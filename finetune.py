"""Fine-tune MERT's last four blocks with train-only epoch selection and fresh refit."""
import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from features import MODEL_ID, MODEL_REVISION
from finetune_model import (FineTunedMERT, capture_reference, checkpoint_dict,
                            read_windows, score_recording, weight_audit)
from train import CLASSES, accuracies, analyze, read_manifest, write_json

SEED = 2026
EFFECTIVE_BATCH = 16


def seed_epoch(epoch=0):
    seed = SEED + epoch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def audio_path(root, row):
    path = (root / row['audio_path']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stem != row['sample_id']:
        raise ValueError(f'Invalid audio path for {row["sample_id"]}: {path}')
    return path


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def optimizer_for(model):
    return torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-3},
    ], weight_decay=0.01)


def encoder_gradient_proof(model):
    blocks = {}
    for index in range(model.first_layer, len(model.backbone.encoder.layers)):
        parameters = [p for name, p in model.backbone.named_parameters()
                      if p.requires_grad and name.startswith(f'encoder.layers.{index}.')]
        gradients = [p.grad for p in parameters if p.grad is not None]
        if len(gradients) != len(parameters) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError(f'Missing or nonfinite pretrained gradients in block {index}')
        norm = sum(float(g.detach().square().sum()) for g in gradients) ** 0.5
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f'No finite nonzero pretrained gradient in block {index}')
        blocks[str(index)] = {'gradient_l2': norm, 'parameter_tensors_with_gradient': len(gradients),
                              'all_gradients_finite': True}
    return {'blocks': blocks, 'measured_before_gradient_clipping': True}


def train_epoch(model, optimizer, rows, root, classes, micro_batch_size, epoch, proof):
    rng = seed_epoch(epoch)
    order = rng.permutation(len(rows)).tolist()
    model.train()
    total_loss, correct = 0.0, 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    for start in range(0, len(order), EFFECTIVE_BATCH):
        group = [rows[i] for i in order[start:start + EFFECTIVE_BATCH]]
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(group), micro_batch_size):
            batch = group[offset:offset + micro_batch_size]
            x = torch.cat([read_windows(audio_path(root, row), rng) for row in batch]).to(model.device)
            y = torch.tensor([classes.index(row['label']) for row in batch], device=model.device)
            logits = model(x)
            loss_sum = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss_sum):
                raise ValueError(f'Nonfinite training loss in epoch {epoch}')
            # Normalize by the actual effective batch, including the final partial batch.
            (loss_sum / len(group)).backward()
            total_loss += float(loss_sum.detach())
            correct += int((logits.detach().argmax(1) == y).sum())
        if proof is None:
            proof = dict(encoder_gradient_proof(model), epoch=epoch, effective_batch_samples=len(group))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = start + len(group)
        if completed % 256 == 0 or completed == len(rows):
            print(f'Epoch {epoch} training: {completed}/{len(rows)} recordings, mean loss {total_loss/completed:.5f}', flush=True)
    return {'train_loss': total_loss / len(rows), 'train_crop_top1': correct / len(rows),
            'train_samples': len(rows), 'optimizer_steps': (len(rows) + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH}, proof


def score_rows(model, rows, root, classes, description):
    scores = []
    model.eval()
    for index, row in enumerate(rows, 1):
        scores.append(score_recording(model, audio_path(root, row)))
        if index % 50 == 0 or index == len(rows):
            print(f'{description}: {index}/{len(rows)} recordings', flush=True)
    scores = np.asarray(scores)
    if scores.shape != (len(rows), 6) or not np.isfinite(scores).all():
        raise ValueError(f'Invalid recording scores: {description}')
    return np.asarray(classes)[np.argsort(-scores, axis=1, kind='stable')]


def phase_signature(context, phase, epochs):
    return hashlib.sha256(json.dumps([context, phase, epochs], sort_keys=True).encode()).hexdigest()


def run_phase(args, dataset, phase, rows, held_out, epochs, context):
    seed_epoch()
    model = FineTunedMERT(args.model_dir, args.device, trainable_layers=4)
    # The shared pretrained loader seeds Torch internally; seed the new task head afterwards.
    seed_epoch()
    for layer in model.head.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
            layer.reset_parameters()
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('The prescribed fine-tuning protocol requires FP32 parameters')
    reference = capture_reference(model)
    optimizer = optimizer_for(model)
    resume_file = args.work_dir / dataset / f'{phase}_resume.pt'
    signature = phase_signature(context, phase, epochs)
    history, proof, start = [], None, 1
    if resume_file.exists():
        saved = torch.load(resume_file, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError(f'Resume protocol differs; preserve or move {resume_file} before starting another experiment')
        if saved['frozen_base_sha256'] != reference['frozen_sha256']:
            raise ValueError(f'Pretrained frozen base differs from the completed epochs in {resume_file}')
        model.restore_backbone(saved['backbone'])
        model.head.load_state_dict(saved['head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history, proof, start = saved['history'], saved['encoder_gradient_proof'], saved['epoch'] + 1
        if [entry['epoch'] for entry in history] != list(range(1, start)) or start > epochs + 1:
            raise ValueError(f'Invalid resume epoch history: {resume_file}')
        print(f'{dataset} {phase}: resumed after epoch {start - 1}', flush=True)
        del saved
    root, classes = args.data_dir / dataset, CLASSES[dataset]
    for epoch in range(start, epochs + 1):
        began = time.monotonic()
        record, proof = train_epoch(model, optimizer, rows, root, classes, args.micro_batch_size, epoch, proof)
        record['epoch'] = epoch
        if held_out:
            ranked = score_rows(model, held_out, root, classes, f'{dataset} epoch {epoch} internal holdout')
            metric = accuracies([r['label'] for r in held_out], ranked)
            record['holdout_top1'], record['holdout_top3'] = metric['top1'], metric['top3']
            record['selection_score'] = metric['top1'] + 0.5 * metric['top3']
        record['elapsed_seconds'] = time.monotonic() - began
        history.append(record)
        saved = checkpoint_dict(model, {'dataset': dataset, 'classes': classes, 'phase': phase})
        saved.update(signature=signature, optimizer=optimizer.state_dict(), epoch=epoch,
                     history=history, encoder_gradient_proof=proof)
        atomic_torch_save(saved, resume_file)
        del saved
        write_json(args.work_dir / dataset / f'{phase}_history.json', history)
        print(json.dumps({'dataset': dataset, 'phase': phase, **record}), flush=True)
    return model, reference, history, proof


def protocol(args):
    return {
        'base_model': MODEL_ID, 'base_revision': MODEL_REVISION, 'seed': SEED,
        'trainable_transformer_blocks': [8, 9, 10, 11], 'trainable_pretrained_parameters': 28351488,
        'precision': 'float32', 'head': 'LayerNorm(1536), Dropout(0.1), Linear(1536,6)',
        'pooling': 'Final hidden-state frame mean and sqrt(population variance + 1e-5), concatenated',
        'training_input': 'One uniformly sampled contiguous 5-second crop per official-train recording per epoch',
        'waveform_normalization': 'Per-window zero mean and divide by sqrt(variance + 1e-7)',
        'inference': 'Six fixed non-overlapping five-second windows covering all 30 seconds; mean of six logits, microbatch 2',
        'optimizer': 'AdamW', 'backbone_lr': 1e-5, 'head_lr': 1e-3, 'weight_decay': 0.01,
        'lr_schedule': 'constant', 'gradient_clip_norm': 1.0, 'effective_batch_size': EFFECTIVE_BATCH,
        'micro_batch_size': args.micro_batch_size, 'max_epochs': args.max_epochs,
        'selection': 'Stratified 20% internal holdout drawn only from official train; maximize Top-1 + 0.5 * Top-3, tie Top-1 then earliest epoch',
        'final_refit': 'Reset original pinned pretrained weights and fresh seeded head; fit all official train for exactly the selected epoch count',
        'official_validation': 'Final reporting only. Its earlier frozen-baseline results were already observed; no fine-tuning setting or submission model is chosen from the official validation results.',
        'submission_choice': 'Use the user-requested fine-tuned model for each task regardless of its validation comparison with the frozen baseline',
        'augmentation': 'Random temporal crop only; no pitch shift, time stretch, source separation, or extra labeled audio',
        'resume': 'Atomic completed-epoch checkpoints; seed 2026 + epoch resets shuffle, crop, and dropout RNG',
    }


def dataset_context(args, dataset, settings):
    manifest = args.data_dir / dataset / 'manifest.csv'
    rows = read_manifest(manifest, dataset)
    training = [r for r in rows if r['split'] == 'train']
    y = [r['label'] for r in training]
    fit, holdout = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(y)), y))
    fitting, held_out = [training[i] for i in fit], [training[i] for i in holdout]
    fit_ids, held_ids = {r['sample_id'] for r in fitting}, {r['sample_id'] for r in held_out}
    if fit_ids & held_ids or fit_ids | held_ids != {r['sample_id'] for r in training}:
        raise ValueError('Internal holdout must partition only official train recordings')
    context = {'dataset': dataset, 'classes': CLASSES[dataset], 'protocol': settings,
               'manifest_sha256': file_sha256(manifest),
               'fit_ids': [r['sample_id'] for r in fitting], 'holdout_ids': [r['sample_id'] for r in held_out]}
    return rows, training, fitting, held_out, context


def run_dataset(args, dataset, settings, rows, training, fitting, held_out, context):
    selection_model, reference, history, selection_proof = run_phase(
        args, dataset, 'selection', fitting, held_out, args.max_epochs, context)
    winner = max(history, key=lambda r: (r['selection_score'], r['holdout_top1'], -r['epoch']))
    selected = {'name': f'MERT last-four-block fine-tuning, {winner["epoch"]} epochs',
                'epochs': winner['epoch'], 'selection_score': winner['selection_score'],
                'holdout_top1': winner['holdout_top1'], 'holdout_top3': winner['holdout_top3']}
    del selection_model, reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({'dataset': dataset, 'selected': selected}), flush=True)
    model, reference, final_history, final_proof = run_phase(
        args, dataset, 'final', training, [], selected['epochs'], context)
    audit = weight_audit(model, reference)
    if audit['trainable_pretrained_parameters'] != settings['trainable_pretrained_parameters']:
        raise ValueError('Unexpected count of updated pretrained parameters')
    selection = {'fit_samples': len(fitting), 'holdout_samples': len(held_out),
                 'fit_ids': context['fit_ids'], 'holdout_ids': context['holdout_ids'],
                 'history': history, 'encoder_gradient_proof': selection_proof}
    final_training = {'n_samples': len(training), 'history': final_history,
                      'encoder_gradient_proof': final_proof}
    metadata = {'dataset': dataset, 'classes': CLASSES[dataset], 'protocol': settings,
                'manifest_sha256': context['manifest_sha256'], 'selected': selected,
                'selection': selection, 'final_training': final_training, 'weight_audit': audit}
    checkpoint_path = args.checkpoint_dir / f'{dataset}_finetuned.pt'
    payload = checkpoint_dict(model, metadata)
    atomic_torch_save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    for section in ['backbone', 'head']:
        if set(loaded[section]) != set(payload[section]) or not all(
                torch.equal(value, loaded[section][key]) for key, value in payload[section].items()):
            raise ValueError(f'Serialized {section} tensors differ from the trained model')
    # Exercise the actual restoration API, not only the serializer.
    model.restore_backbone(loaded['backbone'])
    model.head.load_state_dict(loaded['head'], strict=True)
    if not all(torch.equal(value, loaded['backbone'][key]) for key, value in model.backbone_state().items()):
        raise ValueError('Restored pretrained tensors differ from the saved checkpoint')
    if not all(torch.equal(value.detach().cpu(), loaded['head'][key]) for key, value in model.head.state_dict().items()):
        raise ValueError('Restored head tensors differ from the saved checkpoint')
    del payload, loaded, reference
    validation = [r for r in rows if r['split'] == 'validation']
    testing = [r for r in rows if r['split'] == 'test']
    root, classes = args.data_dir / dataset, CLASSES[dataset]
    validation_ranked = score_rows(model, validation, root, classes, f'{dataset} official validation')
    validation_metrics = analyze([r['label'] for r in validation], validation_ranked, classes, dataset)
    write_json(args.results_dir / f'{dataset}_finetune_validation_predictions.json', {
        row['sample_id']: {'true': row['label'], 'top3': ranking[:3].tolist()}
        for row, ranking in zip(validation, validation_ranked)
    })
    test_ranked = score_rows(model, testing, root, classes, f'{dataset} test')
    predictions = {row['sample_id']: ranking[:3].tolist() for row, ranking in zip(testing, test_ranked)}
    result = dict(classes=classes, selected=selected, selection=selection, final_training=final_training,
                  validation=validation_metrics, weight_audit=audit, checkpoint_roundtrip_exact=True,
                  checkpoint=str(checkpoint_path), checkpoint_sha256=file_sha256(checkpoint_path),
                  protocol_signature=phase_signature(context, 'experiment', args.max_epochs),
                  train_samples=len(training), test_samples=len(testing))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, predictions


def self_check():
    """Exercise real batching/gradient proof on a tiny CPU model, without MERT downloads."""
    import tempfile
    global read_windows

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.encoder = torch.nn.Module()
            self.backbone.encoder.layers = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(8)] + [torch.nn.Linear(3, 3) for _ in range(4)])
            self.first_layer, self.device = 8, torch.device('cpu')
            self.head = torch.nn.Linear(3, 6)

        def forward(self, x):
            for layer in self.backbone.encoder.layers:
                x = layer(x)
            return self.head(x)

    seed_epoch()
    model = Tiny()
    original_reader = read_windows
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for index in range(17):
            path = root / f'A_{index}.wav'
            path.touch()
            rows.append({'sample_id': path.stem, 'audio_path': path.name, 'label': CLASSES['dataset_A'][index % 6]})
        try:
            read_windows = lambda path, rng: torch.from_numpy(rng.normal(size=(1, 3)).astype(np.float32))
            before = {name: p.detach().clone() for name, p in model.backbone.named_parameters()}
            record, proof = train_epoch(model, optimizer_for(model), rows, root, CLASSES['dataset_A'], 4, 1, None)
            assert record['train_samples'] == 17 and record['optimizer_steps'] == 2
            assert np.isfinite(record['train_loss']) and set(proof['blocks']) == {'8', '9', '10', '11'}
            assert all(not torch.equal(before[name], value) for name, value in model.backbone.named_parameters())
            saved = root / 'roundtrip.pt'
            atomic_torch_save({'history': [record], 'proof': proof}, saved)
            assert torch.load(saved, weights_only=True) == {'history': [record], 'proof': proof}
        finally:
            read_windows = original_reader
    print('Fine-tuning CPU self-check passed: effective batching, pretrained gradients/updates, durable save.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--model-dir', default='models/MERT-v1-95M')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--work-dir', type=Path, default=Path('tmp/finetune'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--micro-batch-size', type=int, choices=[1, 2, 4, 8, 16], default=4)
    parser.add_argument('--max-epochs', type=int, default=6)
    parser.add_argument('--dataset', choices=['both', 'A', 'B'], default='both')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not 1 <= args.max_epochs <= 6:
        parser.error('--max-epochs must be between 1 and the prespecified cap of 6')
    settings = protocol(args)
    metrics_path, prediction_path = args.results_dir / 'finetune_metrics.json', args.results_dir / 'finetune_predictions.json'
    report = json.loads(metrics_path.read_text()) if metrics_path.exists() else {'protocol': settings, 'datasets': {}}
    predictions = json.loads(prediction_path.read_text()) if prediction_path.exists() else {}
    if report['protocol'] != settings:
        raise ValueError('Existing fine-tuning results use another protocol; preserve them in a separate output directory')
    for dataset in CLASSES if args.dataset == 'both' else ['dataset_' + args.dataset]:
        rows, training, fitting, held_out, context = dataset_context(args, dataset, settings)
        previous = report['datasets'].get(dataset)
        checkpoint_path = args.checkpoint_dir / f'{dataset}_finetuned.pt'
        if previous and dataset in predictions and checkpoint_path.exists():
            if (previous['protocol_signature'] != phase_signature(context, 'experiment', args.max_epochs)
                    or previous['checkpoint_sha256'] != file_sha256(checkpoint_path)):
                raise ValueError(f'Completed {dataset} artifacts differ from this protocol/checkpoint')
            print(f'{dataset}: completed artifacts already present; retaining recorded official validation results.', flush=True)
            continue
        report['datasets'][dataset], predictions[dataset] = run_dataset(
            args, dataset, settings, rows, training, fitting, held_out, context)
        write_json(metrics_path, report)
        write_json(prediction_path, predictions)
    print('Fine-tuned checkpoints, metrics, and test predictions saved.', flush=True)


if __name__ == '__main__':
    main()
