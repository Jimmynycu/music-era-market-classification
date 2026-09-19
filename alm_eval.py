"""Evaluate frozen Qwen2.5-Omni through a LOCAL llama.cpp audio server.

No fitting, transcript, title, artist, or label is supplied to the model.
Run `python alm_eval.py --self-check` for the parser/metrics check.
"""
import argparse
import base64
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
import urllib.parse
import urllib.request

CLASSES = {
    'dataset_A': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s'],
    'dataset_B': ['Brazil', 'Germany', 'Italy', 'Spain', 'UK', 'US'],
}
CODES = 'ABCDEF'
# Exactly 720 valid rankings: one occurrence of every class, no free-form output.
GRAMMAR = 'root ::= ' + ' | '.join(json.dumps(','.join(p)) for p in itertools.permutations(CODES))
SYSTEM = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def prompt(dataset):
    task = ('Estimate the release decade of this US music recording.' if dataset == 'dataset_A'
            else 'Estimate the release market of this music recording from the 1980s. A release market is where the record was released; it is not necessarily the artist nationality or the language of the vocals.')
    labels = '; '.join(f'{code} = {label}' for code, label in zip(CODES, CLASSES[dataset]))
    return (task + ' Listen to the whole excerpt. Consider instrumentation, rhythm, vocal language or style when audible, recording texture, and production techniques. '
            'The recording may be instrumental and these cues can be ambiguous. Rank all six candidate labels from most to least likely based only on the audio. '
            f'Candidates: {labels}. Return exactly the six different uppercase letter codes in descending confidence order, separated by commas. '
            'Use every code A through F exactly once. Do not write an explanation.')


def parse_ranking(raw, dataset):
    codes = raw.strip().split(',')
    if len(codes) != 6 or set(codes) != set(CODES):
        raise ValueError(f'Invalid ranking: {raw!r}')
    return [CLASSES[dataset][CODES.index(code)] for code in codes]


def metrics(rows, classes):
    matrix = [[0] * len(classes) for _ in classes]
    for row in rows:
        matrix[classes.index(row['label'])][classes.index(row['ranking'][0])] += 1
    n = len(rows)
    return {
        'n_samples': n,
        'top1': sum(r['label'] == r['ranking'][0] for r in rows) / n,
        'top3': sum(r['label'] in r['ranking'][:3] for r in rows) / n,
        'confusion_matrix': matrix,
        'confusion_matrix_units': 'counts; rows=true, columns=predicted',
    }


def request_ranking(path, dataset, endpoint):
    payload = {
        'model': 'qwen2.5-omni-3b',
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [
                {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(path.read_bytes()).decode('ascii'), 'format': 'wav'}},
                {'type': 'text', 'text': prompt(dataset)},
            ]},
        ],
        'temperature': 0, 'seed': 42, 'max_tokens': 32,
        'grammar': GRAMMAR, 'cache_prompt': True,
    }
    req = urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        answer = json.load(response)
    raw = answer['choices'][0]['message']['content']
    return raw, answer.get('timings', {}), answer.get('usage', {})


def metadata():
    return {
        'model': 'Qwen2.5-Omni-3B (frozen Thinker)',
        'model_url': 'https://huggingface.co/Qwen/Qwen2.5-Omni-3B',
        'gguf_url': 'https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF',
        'revision': 'GGUF repository revision 75f1b73b657a50f5092502799457ccb4a4a1f9df',
        'quantization': 'Q4_K_M language model; Q8_0 multimodal projector/audio encoder',
        'engine': 'llama.cpp b11049, commit efa28e950',
        'engine_url': 'https://github.com/ggml-org/llama.cpp',
        'prompt_design': 'One prespecified cue-based prompt; explicit label mapping and ranked constrained output minimize parsing ambiguity and inference cost on the local hardware. The market prompt distinguishes release market from nationality and language. No examples or validation-label feedback are used.',
        'system_prompt': SYSTEM,
        'prompt_A': prompt('dataset_A'), 'prompt_B': prompt('dataset_B'),
        'audio_processing': 'Complete 30-second mono 24-kHz PCM WAV, internally resampled to 16 kHz. This llama.cpp compatibility path adds 30 seconds of zeros and processes two 3000-frame mel chunks, yielding 1500 audio tokens. No external cropping, augmentation, or waveform normalization; preprocessing equivalence to original Transformers is unverified.',
        'implementation_caveat': 'Results describe this particular quantized experimental llama.cpp implementation, not a verified reproduction of the original Transformers model. The audio payload and parsing were checked without new inference or prompt tuning; see alm_integrity_check.json for pinned source references and remaining uncertainty.',
        'parsing': 'GBNF decoding allows exactly the 720 permutations of A,B,C,D,E,F. Strictly require six distinct codes, map to labels, and use first three. All six labels receive ordinal scores 6,5,4,3,2,1 in rank order; these are not calibrated probabilities.',
        'invalid_handling': 'Log raw output and failures. Retry a failed/invalid answer once with the identical input. If both fail, use the prespecified canonical class order (no label access), flag fallback, and include it in all-sample metrics. Connection/server failures halt the run rather than silently replacing the whole evaluation.',
        'decoding': {'temperature': 0, 'seed': 42, 'max_tokens': 32},
        'server_memory': 'Context 2048; one slot; prompt host cache disabled (--cache-ram 0) after initial samples to keep local RAM bounded. Completed outputs were preserved. GPU allocation approximately 4.1 GiB.',
        'runtime_definition': 'Sum of completed per-sample request wall times, including preprocessing and decoding, excluding downloads, model loading, and interruptions. All saved outputs used the local Vulkan GPU server. Concurrent local CUDA feature/test inference and GPU clock state affected speed; a temporary one-element CUDA context maintained active clocks for the final B samples without changing global GPU settings.',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path('data'))
    ap.add_argument('--output-dir', type=Path, default=Path('results'))
    ap.add_argument('--endpoint', default='http://127.0.0.1:8091')
    ap.add_argument('--compute-device', default='local llama.cpp server; see run log')
    ap.add_argument('--split', choices=['train', 'validation'], default='validation')
    ap.add_argument('--dataset', choices=list(CLASSES) + ['both'], default='both')
    ap.add_argument('--limit', type=int, help='Benchmark only; use a separate --output-dir.')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    if args.self_check:
        assert len(list(itertools.permutations(CODES))) == 720
        assert parse_ranking('F,A,B,C,D,E', 'dataset_A')[:3] == ['2010s', '1960s', '1970s']
        for invalid in ['A,A,B,C,D,E', 'A,B,C', 'A,B,C,D,E,G', 'A,B,C,D,E,F.']:
            try:
                parse_ranking(invalid, 'dataset_A')
            except ValueError:
                pass
            else:
                raise AssertionError(invalid)
        m = metrics([{'label': 'B', 'ranking': ['A', 'B', 'C']}], ['A', 'B', 'C'])
        assert m['top1'] == 0 and m['top3'] == 1 and m['confusion_matrix'][1][0] == 1
        print('ALM parser and metric self-check passed.')
        return
    server = urllib.parse.urlsplit(args.endpoint)
    if server.scheme != 'http' or server.hostname not in {'127.0.0.1', 'localhost'} or server.username or server.password:
        ap.error('Only a local inference endpoint is allowed.')
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be positive.')
    if args.limit is not None and args.output_dir == Path('results'):
        ap.error('Benchmarks need a separate --output-dir to preserve full results.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / 'alm_predictions.jsonl'
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            key = (row['dataset'], row['sample_id'])
            if key in done:
                raise ValueError(f'Duplicate saved prediction: {key}')
            done[key] = row
    result = {'metadata': metadata(), 'split': args.split, 'datasets': {}}
    result['metadata']['compute_device'] = args.compute_device
    signal.signal(signal.SIGTERM, request_stop)
    with log.open('a') as out:
        for dataset in (CLASSES if args.dataset == 'both' else [args.dataset]):
            with (args.data_dir / dataset / 'manifest.csv').open(newline='') as f:
                selected = sorted((r for r in csv.DictReader(f) if r['split'] == args.split), key=lambda r: r['sample_id'])
            if args.limit is not None:
                selected = selected[:args.limit]
            if not selected or len({r['sample_id'] for r in selected}) != len(selected):
                raise ValueError('Empty split or duplicate sample IDs.')
            run_rows = []
            prompt_hash = hashlib.sha256((SYSTEM + prompt(dataset)).encode()).hexdigest()
            for number, item in enumerate(selected, 1):
                if STOP_REQUESTED:
                    print('Paused after saving the last complete sample; rerun to resume.', flush=True)
                    return
                key = (dataset, item['sample_id'])
                if key in done:
                    row = done[key]
                    if row['prompt_sha256'] != prompt_hash or row['split'] != args.split:
                        raise ValueError('Saved prediction used a different prompt or split.')
                    run_rows.append(row)
                    continue
                dataset_dir = (args.data_dir / dataset).resolve()
                audio = (dataset_dir / item['audio_path']).resolve()
                if not audio.is_relative_to(dataset_dir) or not audio.is_file() or item['label'] not in CLASSES[dataset]:
                    raise ValueError(f'Missing audio or invalid evaluation label: {audio}')
                began = time.monotonic()
                attempts = []
                for attempt in range(2):
                    raw, timing, usage = request_ranking(audio, dataset, args.endpoint)
                    try:
                        ranking = parse_ranking(raw, dataset)
                        attempts.append({'raw': raw, 'valid': True, 'timing': timing, 'usage': usage})
                        break
                    except ValueError as exc:
                        attempts.append({'raw': raw, 'valid': False, 'error': str(exc)})
                fallback = not attempts[-1]['valid']
                if fallback:
                    ranking = CLASSES[dataset][:]
                row = {
                    'dataset': dataset, 'sample_id': item['sample_id'], 'split': args.split,
                    'label': item['label'], 'ranking': ranking, 'top3': ranking[:3],
                    'scores': {label: 6 - i for i, label in enumerate(ranking)},
                    'prompt_sha256': prompt_hash, 'attempts': attempts, 'fallback': fallback,
                    'compute_device': args.compute_device,
                    'elapsed_seconds': time.monotonic() - began,
                }
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                out.flush()
                os.fsync(out.fileno())
                done[key] = row
                run_rows.append(row)
                print(f'{dataset} {number}/{len(selected)} {item["sample_id"]} {ranking[:3]} {row["elapsed_seconds"]:.1f}s', flush=True)
            result['datasets'][dataset] = {
                'classes': CLASSES[dataset],
                'invalid_outputs': sum(not attempt['valid'] for r in run_rows for attempt in r['attempts']),
                'fallback_samples': sum(r['fallback'] for r in run_rows),
                'elapsed_seconds': sum(r['elapsed_seconds'] for r in run_rows),
                args.split: metrics(run_rows, CLASSES[dataset]),
            }
            dest = args.output_dir / 'alm_metrics.json'
            temp = dest.with_suffix('.json.tmp')
            temp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
            temp.replace(dest)
    print(json.dumps(result['datasets'], indent=2), flush=True)


if __name__ == '__main__':
    main()
