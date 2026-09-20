"""Small offline checks for fresh-run isolation and historical-result scoring."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import alm_eval


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for dataset, classes in alm_eval.CLASSES.items():
            folder = root / 'data' / dataset
            folder.mkdir(parents=True)
            sample_id = dataset[-1] + '_sample'
            (folder / (sample_id + '.wav')).write_bytes(b'fake audio; request is mocked')
            with (folder / 'manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['sample_id', 'split', 'label', 'audio_path'])
                writer.writeheader()
                writer.writerow(dict(sample_id=sample_id, split='validation', label=classes[1],
                                     audio_path=sample_id + '.wav'))
        output = root / 'fresh'
        output.mkdir()  # An existing but empty directory is also a valid fresh run.
        argv = ['alm_eval.py', '--data-dir', str(root / 'data'), '--output-dir', str(output)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(alm_eval, 'request_ranking', return_value=('A,B,C,D,E,F', {}, {})) as request:
            alm_eval.main()
            assert request.call_count == 2
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with patch.object(sys, 'argv', argv), patch.object(alm_eval, 'request_ranking') as request:
            try:
                alm_eval.main()
            except ValueError as error:
                assert 'empty --output-dir' in str(error) and '--score-only' in str(error)
            else:
                raise AssertionError('Existing logs must never be used for new inference.')
            request.assert_not_called()
        stdout = io.StringIO()
        with patch.object(sys, 'argv', argv + ['--score-only']), contextlib.redirect_stdout(stdout), \
                patch.object(alm_eval, 'request_ranking') as request, \
                patch.object(alm_eval.urllib.request, 'urlopen') as network:
            alm_eval.main()
            request.assert_not_called()
            network.assert_not_called()
        result = json.loads(stdout.getvalue())
        assert result['matches_saved_metrics'] is True and result['model_execution_performed'] is False
        for dataset in alm_eval.CLASSES:
            metric = result['datasets'][dataset]['validation']
            assert metric['n_samples'] == 1 and metric['top1'] == 0 and metric['top3'] == 1
            assert metric['confusion_matrix'][1][0] == 1
        assert before == {p.name: p.read_bytes() for p in output.iterdir()}
        # Incomplete historical logs cannot be presented as a complete evaluation.
        log = output / 'alm_predictions.jsonl'
        log.write_text(log.read_text().splitlines()[0] + '\n')
        with patch.object(sys, 'argv', argv + ['--score-only']):
            try:
                alm_eval.main()
            except ValueError as error:
                assert 'exactly cover' in str(error)
            else:
                raise AssertionError('Missing historical samples must be rejected.')
    print('ALM fresh-directory, existing-log rejection and read-only scoring checks passed.')


if __name__ == '__main__':
    main()
