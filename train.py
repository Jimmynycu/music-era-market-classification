"""Select classifiers using training-only CV, then evaluate validation once."""
import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

CLASSES = {
    'dataset_A': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s'],
    'dataset_B': ['US', 'UK', 'Brazil', 'Spain', 'Germany', 'Italy'],
}
RECIPES = {
    'acoustic': '244 acoustic statistics: full-excerpt mean and standard deviation.',
    'mert_middle': 'Mean of MERT layers 5-8; 768 frame means and 768 frame standard deviations.',
    'mert_late': 'Mean of MERT layers 9-12; 768 frame means and 768 frame standard deviations.',
    'mert_all': 'Mean of MERT layers 1-12; 768 frame means and 768 frame standard deviations.',
    'hybrid': 'Concatenated mean MERT layers 1-4, 5-8, 9-12, and acoustic statistics.',
}


def make_vector(features, recipe):
    """The same deterministic recipe is used by training and WAV inference."""
    if recipe not in RECIPES:
        raise ValueError(f'Unknown feature recipe: {recipe}')
    if int(features['feature_version']) != 1:
        raise ValueError('Unsupported feature cache version')
    acoustic = np.asarray(features['acoustic'], dtype=np.float32)
    if acoustic.ndim != 1 or acoustic.size != 244:
        raise ValueError(f'Expected 244 acoustic features, got {acoustic.shape}')
    if recipe == 'acoustic':
        vector = acoustic
    else:
        mert = np.asarray(features['mert'], dtype=np.float32)
        if mert.shape != (13, 1536):
            raise ValueError(f'Expected MERT shape (13, 1536), got {mert.shape}')
        if recipe == 'hybrid':
            vector = np.concatenate([mert[1:5].mean(0), mert[5:9].mean(0), mert[9:13].mean(0), acoustic])
        else:
            start, end = {'mert_middle': (5, 9), 'mert_late': (9, 13), 'mert_all': (1, 13)}[recipe]
            vector = mert[start:end].mean(0)
    if not np.isfinite(vector).all():
        raise ValueError('Nonfinite feature vector')
    return vector


def ranked_labels(estimator, x):
    scores = np.asarray(estimator.decision_function(x))
    if scores.shape != (len(x), 6) or not np.isfinite(scores).all():
        raise ValueError('Expected six finite class scores per sample')
    return np.asarray(estimator.classes_)[np.argsort(-scores, axis=1, kind='stable')]


def accuracies(truth, ranked):
    truth = np.asarray(truth)
    return {
        'top1': float(np.mean(ranked[:, 0] == truth)),
        'top3': float(np.mean(np.any(ranked[:, :3] == truth[:, None], axis=1))),
    }


def analyze(truth, ranked, classes, dataset):
    result = accuracies(truth, ranked)
    matrix = confusion_matrix(truth, ranked[:, 0], labels=classes)
    errors = int(matrix.sum() - np.trace(matrix))
    confusions = [
        {'true': classes[i], 'predicted': classes[j], 'count': int(matrix[i, j])}
        for i in range(6) for j in range(6) if i != j and matrix[i, j]
    ]
    confusions.sort(key=lambda row: -row['count'])
    result.update(
        n_samples=len(truth), confusion_matrix=matrix.tolist(), confusion_matrix_kind='counts',
        total_errors=errors, largest_confusions=confusions[:8],
        per_class_accuracy={label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None
                            for i, label in enumerate(classes)},
    )
    if dataset == 'dataset_A':
        adjacent = int(sum(matrix[i, j] for i in range(6) for j in range(6) if abs(i-j) == 1))
        result.update(adjacent_errors=adjacent, adjacent_error_fraction=adjacent/errors if errors else 0.0)
    return result


def candidates():
    for recipe in RECIPES:
        family = 'acoustic' if recipe == 'acoustic' else 'hybrid' if recipe == 'hybrid' else 'mert'
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            yield dict(name=f'{recipe}: ridge alpha={alpha:g}', feature_recipe=recipe,
                       feature_family=family, classifier='ridge', parameters={'alpha': alpha})
        for c in [0.1, 1.0, 10.0, 100.0]:
            yield dict(name=f'{recipe}: RBF SVC C={c:g}', feature_recipe=recipe,
                       feature_family=family, classifier='rbf_svc', parameters={'C': c, 'gamma': 'scale'})


def estimator_for(candidate):
    if candidate['classifier'] == 'ridge':
        model = RidgeClassifier(**candidate['parameters'], solver='cholesky')
    else:
        model = SVC(**candidate['parameters'], kernel='rbf', decision_function_shape='ovr', cache_size=1024)
    return make_pipeline(StandardScaler(), Normalizer(norm='l2'), model)


def read_manifest(path, dataset):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        sample_id = row['sample_id']
        if sample_id in seen or Path(sample_id).name != sample_id or not sample_id.startswith(dataset[-1] + '_'):
            raise ValueError(f'Duplicate or invalid sample ID: {sample_id}')
        seen.add(sample_id)
        if row['split'] not in ['train', 'validation', 'test']:
            raise ValueError(f'Invalid split: {row}')
        if row['split'] != 'test' and row['label'] not in CLASSES[dataset]:
            raise ValueError(f'Invalid training/validation label: {row}')
    return rows


def load_features(rows, cache_dir, recipes):
    matrices = {recipe: [] for recipe in recipes}
    for row in rows:
        with np.load(cache_dir / (row['sample_id'] + '.npz'), allow_pickle=False) as feature:
            for recipe in recipes:
                matrices[recipe].append(make_vector(feature, recipe))
    return {recipe: np.stack(vectors) for recipe, vectors in matrices.items()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def train_dataset(dataset, args):
    began = time.time()
    rows = read_manifest(args.data_dir / dataset / 'manifest.csv', dataset)
    training = [r for r in rows if r['split'] == 'train']
    truth = np.asarray([r['label'] for r in training])
    matrices = load_features(training, args.cache_dir / dataset, RECIPES)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=2026).split(np.zeros(len(truth)), truth))
    results = []
    for candidate in candidates():
        features = matrices[candidate['feature_recipe']]
        ranked = np.empty((len(training), 6), dtype=truth.dtype)
        for fit, held_out in folds:
            estimator = estimator_for(candidate).fit(features[fit], truth[fit])
            ranked[held_out] = ranked_labels(estimator, features[held_out])
        metric = accuracies(truth, ranked)
        result = dict(candidate, **metric, selection_score=metric['top1'] + 0.5 * metric['top3'])
        results.append(result)
        write_json(args.results_dir / f'{dataset}_cv_progress.json', results)
        print(json.dumps({'dataset': dataset, 'candidate': len(results), **result}), flush=True)
    selected = max(results, key=lambda result: (result['selection_score'], result['top1']))
    recipe = selected['feature_recipe']
    estimator = estimator_for(selected).fit(matrices[recipe], truth)
    selected_info = dict(selected, cv_top1=selected['top1'], cv_top3=selected['top3'],
                         description=RECIPES[recipe], feature_dimensions=matrices[recipe].shape[1])
    checkpoint = dict(format_version=1, dataset=dataset, classes=CLASSES[dataset], feature_recipe=recipe,
                      estimator=estimator, selection=selected_info, training_samples=len(training),
                      random_seed=2026, feature_version=1)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint_dir / f'{dataset}.tmp.joblib'
    joblib.dump(checkpoint, temporary, compress=3)
    temporary.replace(args.checkpoint_dir / f'{dataset}.joblib')
    # All comparison candidates are fixed using training CV before evaluating validation.
    comparison_choices = {
        family: max((r for r in results if r['feature_family'] == family),
                    key=lambda r: (r['selection_score'], r['top1']))
        for family in ['acoustic', 'mert']
    }
    # Validation features are first accessed after model selection and fitting.
    validation = [r for r in rows if r['split'] == 'validation']
    validation_recipes = set([recipe] + [r['feature_recipe'] for r in comparison_choices.values()])
    validation_matrices = load_features(validation, args.cache_dir / dataset, validation_recipes)
    validation_features = validation_matrices[recipe]
    validation_ranked = ranked_labels(estimator, validation_features)
    metrics = analyze([r['label'] for r in validation], validation_ranked, CLASSES[dataset], dataset)
    comparisons = {}
    for family, choice in comparison_choices.items():
        comparison_recipe = choice['feature_recipe']
        comparison_model = estimator if choice['name'] == selected['name'] else estimator_for(choice).fit(matrices[comparison_recipe], truth)
        comparison_ranked = ranked_labels(comparison_model, validation_matrices[comparison_recipe])
        comparisons[family] = {
            'selected': dict(choice, cv_top1=choice['top1'], cv_top3=choice['top3'], description=RECIPES[comparison_recipe]),
            'validation': analyze([r['label'] for r in validation], comparison_ranked, CLASSES[dataset], dataset),
        }
    write_json(args.results_dir / f'{dataset}_validation_predictions.json', {
        row['sample_id']: {'true': row['label'], 'top3': rank[:3].tolist()}
        for row, rank in zip(validation, validation_ranked)
    })
    testing = [r for r in rows if r['split'] == 'test']
    test_features = load_features(testing, args.cache_dir / dataset, [recipe])[recipe]
    test_ranked = ranked_labels(estimator, test_features)
    predictions = {row['sample_id']: rank[:3].tolist() for row, rank in zip(testing, test_ranked)}
    details = dict(classes=CLASSES[dataset], selected=selected_info, cv_results=results,
                   validation=metrics, comparisons=comparisons, train_samples=len(training), test_samples=len(testing),
                   training_seconds=round(time.time() - began, 2))
    return details, predictions


def self_check():
    features = dict(mert=np.repeat(np.arange(13, dtype=np.float32)[:, None], 1536, axis=1),
                    acoustic=np.ones(244, dtype=np.float32), feature_version=np.array(1))
    assert make_vector(features, 'acoustic').shape == (244,)
    assert np.all(make_vector(features, 'mert_middle') == 6.5)
    assert np.all(make_vector(features, 'mert_late') == 10.5)
    assert np.all(make_vector(features, 'mert_all') == 6.5)
    hybrid = make_vector(features, 'hybrid')
    assert hybrid.shape == (4852,) and hybrid[0] == 2.5 and hybrid[-1] == 1
    labels = CLASSES['dataset_A']
    ranked = np.asarray([labels, labels[1:] + labels[:1], labels[3:] + labels[:3]])
    result = analyze(labels[:3], ranked, labels, 'dataset_A')
    assert result['top1'] == 2/3 and result['top3'] == 2/3
    assert result['adjacent_errors'] == 1 and result['total_errors'] == 1
    rng = np.random.default_rng(2026)
    x = np.repeat(np.eye(6), 5, axis=0) + rng.normal(0, 0.01, (30, 6))
    y = np.repeat(labels, 5)
    estimator = estimator_for(next(candidates())).fit(x, y)
    assert np.all(ranked_labels(estimator, x)[:, 0] == y)
    features['mert'][0, 0] = np.nan
    features['acoustic'][0] = np.nan
    try:
        make_vector(features, 'acoustic')
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite inputs must be rejected')
    print('Training self-check passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--cache-dir', type=Path, default=Path('cache'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    # The local Haswell OpenBLAS float32 Cholesky kernel crashes with four threads.
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.threads < 1:
        parser.error('--threads must be positive')
    report = dict(protocol={
        'fit_split': 'train', 'selection': 'training-only stratified three-fold cross-validation',
        'selection_score': 'top1 + 0.5 * top3; ties use top1 then fixed candidate order',
        'random_seed': 2026, 'preprocessing': 'StandardScaler fitted within each training fold, then per-recording L2 normalization',
        'validation_use': 'single final evaluation after selecting and refitting on full training split',
        'augmentation': 'none', 'threads': args.threads,
    }, datasets={})
    predictions = {}
    with threadpool_limits(limits=args.threads):
        for dataset in CLASSES:
            report['datasets'][dataset], predictions[dataset] = train_dataset(dataset, args)
            write_json(args.results_dir / 'metrics.json', report)
            write_json(args.results_dir / 'predictions.json', predictions)
    print(json.dumps({dataset: report['datasets'][dataset]['validation'] for dataset in CLASSES}, indent=2))


if __name__ == '__main__':
    main()
