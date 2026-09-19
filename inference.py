"""Produce the assignment top-three JSON from grader-local WAV directories."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from features import Extractor
from train import CLASSES, make_vector, ranked_labels


def locate_samples(root, prefix):
    root = Path(root).resolve()
    samples = {}
    manifest = root / 'manifest.csv'
    if manifest.exists():
        with manifest.open(newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['split'] == 'test']
        paths = [root / r['audio_path'] for r in rows]
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate test IDs in {manifest}')
        for row, path in zip(rows, paths):
            if not row['sample_id'].startswith(prefix):
                raise ValueError(f'Wrong dataset sample ID in {manifest}: {row["sample_id"]}')
            if not path.resolve().is_relative_to(root) or not path.is_file() or path.stem != row['sample_id']:
                raise ValueError(f'Invalid test audio path: {path}')
    else:
        paths = sorted(root.rglob('*.wav'))
    for path in paths:
        if not path.stem.startswith(prefix):
            continue
        if path.stem in samples:
            raise ValueError(f'Duplicate sample ID {path.stem} in {root}')
        samples[path.stem] = path
    if not samples:
        raise ValueError(f'No {prefix} WAV files under {root}')
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-a', required=True, type=Path, help='Dataset A root with manifest.csv, or test-only WAV directory')
    p.add_argument('--dataset-b', required=True, type=Path, help='Dataset B root with manifest.csv, or test-only WAV directory')
    p.add_argument('--checkpoint-dir', type=Path, default=Path(__file__).parent / 'checkpoints')
    p.add_argument('--model-dir', help='Optional local MERT snapshot; default downloads pinned public model')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--method', choices=['finetuned', 'frozen'], default='finetuned',
                   help='Submitted fine-tuned models (default), or the frozen-feature baseline')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    extractor = Extractor(args.model_dir, args.device) if args.method == 'frozen' else None
    predictions = {}
    for dataset, root, prefix in [('dataset_A', args.dataset_a, 'A_'), ('dataset_B', args.dataset_b, 'B_')]:
        if args.method == 'finetuned':
            from finetune_model import load_finetuned, score_recording
            model, checkpoint = load_finetuned(args.checkpoint_dir / f'{dataset}_finetuned.pt',
                                               args.model_dir, args.device)
            if (checkpoint['metadata']['dataset'] != dataset
                    or checkpoint['metadata']['classes'] != CLASSES[dataset]):
                raise ValueError(f'Wrong dataset or class order in {dataset} checkpoint')
        else:
            checkpoint = joblib.load(args.checkpoint_dir / f'{dataset}.joblib')
            if checkpoint['dataset'] != dataset or set(checkpoint['estimator'].classes_) != set(CLASSES[dataset]):
                raise ValueError(f'Wrong dataset or class labels in {dataset} checkpoint')
        samples = locate_samples(root, prefix)
        predictions[dataset] = {}
        for index, (sample_id, path) in enumerate(samples.items(), 1):
            if args.method == 'finetuned':
                scores = score_recording(model, path)
                predictions[dataset][sample_id] = np.asarray(CLASSES[dataset])[np.argsort(-scores, kind='stable')[:3]].tolist()
            else:
                feature = make_vector(extractor(path), checkpoint['feature_recipe'])
                predictions[dataset][sample_id] = ranked_labels(checkpoint['estimator'], feature.reshape(1, -1))[0, :3].tolist()
            print(f'{dataset} {index}/{len(samples)} {sample_id}', flush=True)
        if args.method == 'finetuned':
            del model, checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(predictions, indent=2) + '\n')
    temp.replace(args.output)


if __name__ == '__main__':
    main()
