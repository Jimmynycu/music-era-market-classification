"""Partially fine-tuned MERT and the shared full-recording inference path."""
import hashlib
import json

import numpy as np
import torch
from torch import nn

from features import Extractor, MODEL_ID, MODEL_REVISION, read_audio


class FineTunedMERT(nn.Module):
    def __init__(self, model_dir=None, device='auto', trainable_layers=4):
        super().__init__()
        if not 1 <= trainable_layers <= 12:
            raise ValueError('trainable_layers must be 1..12')
        # Reuse the strict loader, including the legacy positional-weight mapping.
        self.backbone = Extractor(model_dir, device).model
        self.backbone.feature_extractor._freeze_parameters()
        self.backbone.config.apply_spec_augment = False
        self.backbone.config.layerdrop = 0.0
        self.trainable_layers = trainable_layers
        self.first_layer = len(self.backbone.encoder.layers) - trainable_layers
        for layer in self.backbone.encoder.layers[self.first_layer:]:
            layer.requires_grad_(True)
        self.head = nn.Sequential(nn.LayerNorm(1536), nn.Dropout(0.1), nn.Linear(1536, 6)).to(self.device)
        self.eval()

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        # The frozen prefix must not inject dropout into its fixed representations.
        self.backbone.feature_extractor.eval()
        self.backbone.feature_projection.eval()
        self.backbone.encoder.pos_conv_embed.eval()
        self.backbone.encoder.dropout.eval()
        self.backbone.encoder.layer_norm.eval()
        for layer in self.backbone.encoder.layers[:self.first_layer]:
            layer.eval()
        return self

    def forward(self, normalized_waveforms):
        h = self.backbone(normalized_waveforms, output_hidden_states=False,
                          output_attentions=False).last_hidden_state
        pooled = torch.cat((h.mean(dim=1), (h.var(dim=1, unbiased=False) + 1e-5).sqrt()), dim=1)
        return self.head(pooled)

    def backbone_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.backbone.named_parameters()
                if value.requires_grad}

    def restore_backbone(self, state):
        parameters = {name: value for name, value in self.backbone.named_parameters() if value.requires_grad}
        if set(state) != set(parameters):
            raise ValueError('Fine-tuned backbone keys do not match selected transformer blocks')
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid fine-tuned tensor: {name}')
                parameters[name].copy_(value)


def read_windows(path, rng=None):
    audio = read_audio(path)
    length = 5 * 24000
    if rng is not None:
        start = int(rng.integers(max(1, len(audio) - length + 1)))
        windows = [audio[start:start + length]]
    else:
        windows = [audio[start:start + length] for start in range(0, len(audio), length)]
    windows = np.stack([np.pad(x, (0, length - len(x))) for x in windows])
    windows = (windows - windows.mean(axis=1, keepdims=True)) / np.sqrt(windows.var(axis=1, keepdims=True) + 1e-7)
    return torch.from_numpy(windows.astype(np.float32, copy=False))


@torch.inference_mode()
def score_recording(model, path):
    model.eval()
    windows = read_windows(path)
    logits = torch.cat([model(windows[start:start + 2].to(model.device)).cpu()
                        for start in range(0, len(windows), 2)])
    scores = logits.mean(dim=0).numpy()
    if scores.shape != (6,) or not np.isfinite(scores).all():
        raise ValueError(f'Invalid scores: {path}')
    return scores


def frozen_digest(model):
    trainable = {name for name, value in model.backbone.named_parameters() if value.requires_grad}
    digest = hashlib.sha256()
    for name, value in sorted(model.backbone.state_dict().items()):
        if name not in trainable:
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_reference(model):
    return {'frozen_sha256': frozen_digest(model), 'trainable': model.backbone_state()}


def weight_audit(model, reference):
    current = model.backbone_state()
    blocks = {}
    for index in range(model.first_layer, len(model.backbone.encoder.layers)):
        differences = [value - reference['trainable'][name] for name, value in current.items()
                       if name.startswith(f'encoder.layers.{index}.')]
        if not all(torch.isfinite(delta).all() for delta in differences):
            raise ValueError(f'Nonfinite weights in block {index}')
        changed = sum(int(torch.count_nonzero(delta)) for delta in differences)
        if not changed:
            raise ValueError(f'Pretrained block {index} did not change')
        blocks[str(index)] = {'changed_elements': changed,
                              'delta_l2': sum(float(delta.double().square().sum()) for delta in differences) ** 0.5,
                              'max_abs_delta': max(float(delta.abs().max()) for delta in differences)}
    digest = frozen_digest(model)
    if digest != reference['frozen_sha256']:
        raise ValueError('Frozen pretrained weights changed')
    return {'blocks': blocks, 'frozen_sha256_before': reference['frozen_sha256'],
            'frozen_sha256_after': digest, 'frozen_weights_unchanged': True,
            'trainable_pretrained_parameters': sum(v.numel() for v in current.values())}


def checkpoint_dict(model, metadata):
    return {'format_version': 1, 'base_model': MODEL_ID, 'base_revision': MODEL_REVISION,
            'frozen_base_sha256': frozen_digest(model),
            'trainable_layers': model.trainable_layers, 'backbone': model.backbone_state(),
            'head': {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()},
            'metadata': metadata}


def load_finetuned(path, model_dir=None, device='auto'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['format_version'] != 1 or checkpoint['base_model'] != MODEL_ID
            or checkpoint['base_revision'] != MODEL_REVISION):
        raise ValueError('Unsupported fine-tuned checkpoint or pretrained base revision')
    model = FineTunedMERT(model_dir, device, checkpoint['trainable_layers'])
    if frozen_digest(model) != checkpoint['frozen_base_sha256']:
        raise ValueError('Local pretrained base does not match the fine-tuned checkpoint')
    model.restore_backbone(checkpoint['backbone'])
    if not all(torch.isfinite(value).all() for value in checkpoint['head'].values()):
        raise ValueError('Nonfinite classification head')
    model.head.load_state_dict(checkpoint['head'], strict=True)
    model.eval()
    return model, checkpoint
