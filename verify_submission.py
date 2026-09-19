"""Check prediction coverage, distinct allowed labels, and optional exact reproduction."""
import argparse
import csv
import json
from pathlib import Path

LABELS = {
    'dataset_A': {'1960s', '1970s', '1980s', '1990s', '2000s', '2010s'},
    'dataset_B': {'US', 'UK', 'Brazil', 'Spain', 'Germany', 'Italy'},
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def validate(prediction_file, data_dir):
    predictions = json.loads(Path(prediction_file).read_text(), object_pairs_hook=unique_object)
    if not isinstance(predictions, dict) or set(predictions) != set(LABELS):
        raise ValueError('Expected exactly dataset_A and dataset_B')
    counts = {}
    for dataset, allowed in LABELS.items():
        with (Path(data_dir) / dataset / 'manifest.csv').open(newline='') as f:
            expected_ids = [r['sample_id'] for r in csv.DictReader(f) if r['split'] == 'test']
        expected = set(expected_ids)
        if not expected or len(expected) != len(expected_ids):
            raise ValueError(f'{dataset}: empty or duplicate test IDs in manifest')
        if not isinstance(predictions[dataset], dict) or set(predictions[dataset]) != expected:
            raise ValueError(f'{dataset}: missing or extra test IDs')
        for sample_id, labels in predictions[dataset].items():
            if not isinstance(labels, list) or len(labels) != 3 or any(not isinstance(x, str) for x in labels):
                raise ValueError(f'{sample_id}: expected three label strings')
            if len(set(labels)) != 3 or not set(labels) <= allowed:
                raise ValueError(f'{sample_id}: duplicate or invalid labels')
        counts[dataset] = len(expected)
    return predictions, counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--reproduced', type=Path)
    args = parser.parse_args()
    predictions, counts = validate(args.predictions, args.data_dir)
    if args.reproduced:
        reproduced, _ = validate(args.reproduced, args.data_dir)
        if predictions != reproduced:
            raise ValueError('Regenerated predictions differ')
    print(json.dumps({'valid': True, 'test_samples': counts, 'exact_reproduction_checked': bool(args.reproduced)}))
