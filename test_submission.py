"""Runnable parser, grader-path, and checkpoint checks; no Torch or downloads."""
import csv
import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from inference import locate_samples
from train import CLASSES, candidates, estimator_for, ranked_labels, self_check
from verify_submission import validate


def rejects(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted')


def main():
    with threadpool_limits(limits=1):
        self_check()
        x = np.repeat(np.eye(6), 4, axis=0)
        y = np.repeat(CLASSES['dataset_B'], 4)
        for classifier in ['ridge', 'rbf_svc']:
            choice = next(c for c in candidates() if c['classifier'] == classifier)
            estimator = estimator_for(choice).fit(x, y)
            before = ranked_labels(estimator, x)
            checkpoint = io.BytesIO()
            joblib.dump({'estimator': estimator, 'feature_recipe': choice['feature_recipe']}, checkpoint)
            checkpoint.seek(0)
            assert np.array_equal(before, ranked_labels(joblib.load(checkpoint)['estimator'], x))
            assert np.all(before[:, 0] == y)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        prediction = root / 'predictions.json'
        valid = {}
        for dataset, labels in CLASSES.items():
            folder = root / dataset
            (folder / 'audio').mkdir(parents=True)
            prefix = dataset[-1] + '_'
            rows = []
            for split in ['train', 'validation', 'test']:
                sample_id = prefix + split
                audio_path = 'audio/' + sample_id + '.wav'
                (folder / audio_path).touch()
                rows.append(dict(sample_id=sample_id, split=split, audio_path=audio_path))
            with (folder / 'manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['sample_id', 'split', 'audio_path'])
                writer.writeheader()
                writer.writerows(rows)
            assert set(locate_samples(folder, prefix)) == {prefix + 'test'}
            test_only = root / (dataset + '_test')
            test_only.mkdir()
            (test_only / (prefix + 'test.wav')).touch()
            assert set(locate_samples(test_only, prefix)) == {prefix + 'test'}
            valid[dataset] = {prefix + 'test': labels[:3]}
        prediction.write_text(json.dumps(valid))
        assert validate(prediction, root)[1] == {'dataset_A': 1, 'dataset_B': 1}
        for bad in [[], {'dataset_A': [], 'dataset_B': {}},
                    {**valid, 'dataset_A': {'A_test': ['1960s'] * 3}},
                    {**valid, 'dataset_A': {'A_test': ['1960s', '1970s', 'INVALID']}},
                    {**valid, 'dataset_A': {}}]:
            prediction.write_text(json.dumps(bad))
            rejects(lambda: validate(prediction, root))
        prediction.write_text('{"dataset_A": {}, "dataset_A": {}, "dataset_B": {}}')
        rejects(lambda: validate(prediction, root))
        prediction.write_text(json.dumps(valid))
        manifest = root / 'dataset_A' / 'manifest.csv'
        original = manifest.read_text()
        manifest.write_text(original + 'A_test,test,audio/A_test.wav\n')
        rejects(lambda: locate_samples(root / 'dataset_A', 'A_'))
        rejects(lambda: validate(prediction, root))
        manifest.write_text(original.replace('audio/A_test.wav', '../dataset_B/audio/B_test.wav'))
        rejects(lambda: locate_samples(root / 'dataset_A', 'A_'))
        manifest.write_text(original.replace('A_test,test,audio/A_test.wav', 'B_test,test,../dataset_B/audio/B_test.wav'))
        rejects(lambda: locate_samples(root / 'dataset_A', 'A_'))
    print('Submission integration checks passed: parser, paths, Ridge/SVC checkpoint roundtrip.')


if __name__ == '__main__':
    main()
