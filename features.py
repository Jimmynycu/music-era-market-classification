"""Deterministic full-excerpt MERT and acoustic features; no labels are read."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = 'm-a-p/MERT-v1-95M'
MODEL_REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
FEATURE_VERSION = 1


def read_audio(path):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Empty or nonfinite audio: {path}')
    audio = audio.mean(axis=1)
    if sr != 24000:
        from scipy.signal import resample_poly
        from math import gcd
        div = gcd(sr, 24000)
        audio = resample_poly(audio, 24000 // div, sr // div).astype(np.float32)
    return audio


def acoustic_features(audio):
    import librosa
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    power = spectrum ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=24000, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=1.0)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)
    blocks = [
        ('logmel', logmel), ('mfcc', mfcc),
        ('mfcc_delta', librosa.feature.delta(mfcc)),
        ('chroma', librosa.feature.chroma_stft(S=power, sr=24000, tuning=0)),
        ('centroid', librosa.feature.spectral_centroid(S=spectrum, sr=24000)),
        ('bandwidth', librosa.feature.spectral_bandwidth(S=spectrum, sr=24000)),
        ('rolloff', librosa.feature.spectral_rolloff(S=spectrum, sr=24000)),
        ('flatness', librosa.feature.spectral_flatness(S=spectrum)),
        ('rms', librosa.feature.rms(S=spectrum, frame_length=2048)),
        ('zcr', librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)),
    ]
    values, names = [], []
    for name, block in blocks:
        for stat, v in [('mean', block.mean(axis=1)), ('std', block.std(axis=1))]:
            values.extend(v)
            names.extend(f'{name}_{i}_{stat}' for i in range(len(v)))
    return np.array(values, dtype=np.float32), np.array(names)


class Extractor:
    def __init__(self, model_dir=None, device='auto', batch_size=1):
        import torch
        from transformers import AutoModel
        torch.set_num_threads(2)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        torch.manual_seed(0)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        kwargs = {} if model_dir else {'revision': MODEL_REVISION}
        if model_dir:
            weight_path = Path(model_dir) / 'pytorch_model.bin'
        else:
            from huggingface_hub import hf_hub_download
            weight_path = hf_hub_download(MODEL_ID, 'pytorch_model.bin', revision=MODEL_REVISION)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
        # Preserve the original positional-convolution weights under PyTorch's new names.
        for old, new in [('weight_g', 'parametrizations.weight.original0'),
                         ('weight_v', 'parametrizations.weight.original1')]:
            prefix = 'encoder.pos_conv_embed.conv.'
            if prefix + old in state:
                state[prefix + new] = state.pop(prefix + old)
        self.model, loading = AutoModel.from_pretrained(
            model_dir or MODEL_ID, state_dict=state, output_loading_info=True,
            trust_remote_code=True, **kwargs)
        if any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise RuntimeError(f'Incomplete pretrained weight loading: {loading}')
        del state
        self.model.eval().requires_grad_(False).to(self.device)
        self.batch_size = batch_size

    def __call__(self, path):
        import torch
        audio = read_audio(path)
        # MERT was pretrained on five-second contexts. Use every frame, in order.
        n = 5 * 24000
        segments = [audio[i:i+n] for i in range(0, len(audio), n)]
        segments = [np.pad(x, (0, n-len(x))) for x in segments]
        segments = np.stack(segments)
        segments = (segments - segments.mean(axis=1, keepdims=True)) / np.sqrt(segments.var(axis=1, keepdims=True) + 1e-7)
        sums = squares = None
        count = 0
        with torch.inference_mode():
            for start in range(0, len(segments), self.batch_size):
                x = torch.from_numpy(segments[start:start+self.batch_size]).to(self.device)
                out = self.model(x, output_hidden_states=True)
                h = torch.stack(out.hidden_states).double()
                s, ss = h.sum(dim=(1, 2)), (h*h).sum(dim=(1, 2))
                sums = s if sums is None else sums + s
                squares = ss if squares is None else squares + ss
                count += h.shape[1] * h.shape[2]
            mean = sums / count
            std = (squares / count - mean.square()).clamp_min(0).sqrt()
            mert = torch.cat([mean, std], dim=1).float().cpu().numpy()
        acoustic, names = acoustic_features(audio)
        if mert.shape != (13, 1536) or not np.isfinite(mert).all() or not np.isfinite(acoustic).all():
            raise ValueError(f'Invalid features: {path}')
        return dict(mert=mert, acoustic=acoustic, acoustic_names=names,
                    feature_version=np.array(FEATURE_VERSION), audio_frames=np.array(len(audio)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--cache-dir', type=Path, default=Path('cache'))
    p.add_argument('--model-dir', default='models/MERT-v1-95M')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    files = sorted(args.data_dir.rglob('*.wav'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f'No WAV files under {args.data_dir}')
    extractor = Extractor(args.model_dir, args.device, args.batch_size)
    import torch
    print(json.dumps({'event': 'start', 'model': MODEL_ID, 'revision': MODEL_REVISION,
                      'device': extractor.device, 'torch': torch.__version__,
                      'batch_size': args.batch_size, 'dtype': 'float32'}), flush=True)
    begin, done = time.time(), 0
    for i, path in enumerate(files, 1):
        dataset = 'dataset_A' if path.stem.startswith('A_') else 'dataset_B' if path.stem.startswith('B_') else None
        if not dataset:
            raise ValueError(f'Unexpected sample ID: {path}')
        target = args.cache_dir / dataset / (path.stem + '.npz')
        if target.exists():
            with np.load(target) as existing:
                if int(existing['feature_version']) == FEATURE_VERSION:
                    continue
        features = extractor(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **features)
        temp.replace(target)
        done += 1
        elapsed = time.time() - begin
        print(json.dumps({'completed': i, 'total': len(files), 'id': path.stem,
                          'seconds_per_new_sample': round(elapsed/done, 3),
                          'peak_gpu_allocated_mb': round(torch.cuda.max_memory_allocated()/2**20, 1)
                          if extractor.device == 'cuda' else 0}), flush=True)


if __name__ == '__main__':
    main()
