"""Reconstruct training-only history probes from saved arrays and classifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary = json.loads((args.runs / 'summary.json').read_text())
    assert not summary['validation_evaluated'] and not summary['test_evaluated'] and not summary['policy_training']
    for name, key in [('models.pkl', 'model_sha256'), ('generated.npz', 'generated_sha256')]:
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == summary[key]
    generated = np.load(args.runs / 'generated.npz')
    x, target, current, days = [generated[name] for name in ('x', 'target', 'current', 'row_days')]
    assert x.shape == (89300, 18) and target.shape == current.shape == (89300, 2)
    assert np.isfinite(x).all() and np.all(np.abs(x[:, 17]) <= 1)
    bounds = summary.get('action_bounds', [-1, 1])
    assert np.all(x[:, 17] >= bounds[0]) and np.all(x[:, 17] <= bounds[1])
    fit, held = summary['fit_days'], summary['held_training_days']
    assert len(fit) == 158 and len(held) == 30 and set(fit).isdisjoint(held)
    np.testing.assert_array_equal(days, np.tile(np.array((fit + held) * 5), 95))
    shaped = x.reshape(95, 940, 18)
    columns = [0, 1, 2, 3, 4, 10]
    np.testing.assert_array_equal(shaped[0, :, 11:17], shaped[0][:, columns])
    np.testing.assert_array_equal(shaped[1:, :, 11:17], shaped[:-1][:, :, columns])
    np.testing.assert_allclose(shaped[1:, :, 10], (shaped[:-1, :, 17] + 1) / 2, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(current, np.c_[x[:, 0] > 4, x[:, 2:5].max(-1) > 2])
    repeated = x.copy()
    repeated[:, 11:17] = repeated[:, columns]
    rows = np.isin(days, held)
    assert int(rows.sum()) == 14250
    with (args.runs / 'models.pkl').open('rb') as stream:
        models = pickle.load(stream)
    probabilities = np.load(args.runs / 'held_probabilities.npz')
    maximum = 0.
    errors = {}
    for name in ('current', 'history', 'repeated_current'):
        entry = models[name]
        cols = entry['columns']
        np.testing.assert_array_equal(cols, np.r_[np.arange(11), 17] if name == 'current' else np.arange(18))
        features = repeated if name == 'repeated_current' else x
        values = []
        assert len(entry['models']) == 5
        for member in entry['models']:
            for model in member:
                assert model.n_iter_ == 200 and model.class_weight is None and not model.early_stopping
            values.append(np.stack([model.predict_proba(features[rows][:, cols])[:, 1] for model in member], -1))
        prediction = np.mean(values, 0)
        np.testing.assert_array_equal(prediction, probabilities[name])
        maximum = max(maximum, float(np.abs(prediction - probabilities[name]).max()))
        errors[name] = {}
        for j, label in enumerate(('nh4', 'do')):
            for subset, mask in [('all_steps', np.ones(int(rows.sum()), bool)), ('currently_nonviolating', ~current[rows, j])]:
                p, y = prediction[mask, j], target[rows, j][mask]
                metrics = summary['metrics'][name][label][subset]
                assert len(y) == metrics['count'] and int(y.sum()) == metrics['positives']
                np.testing.assert_allclose(np.mean((p - y) ** 2), metrics['brier'], rtol=1e-12)
                np.testing.assert_allclose(average_precision_score(y, p), metrics['average_precision'], rtol=1e-12)
                np.testing.assert_allclose(roc_auc_score(y, p), metrics['roc_auc'], rtol=1e-12)
            errors[name][label] = np.array([np.mean((prediction[days[rows] == day, j] - target[rows, j][days[rows] == day]) ** 2)
                                          for day in held])
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, 30, (10000, 6))
    indices = ((starts[:, :, None] + np.arange(5)) % 30).reshape(10000, 30)
    for candidate, baseline in [('history', 'current'), ('history', 'repeated_current'), ('repeated_current', 'current')]:
        for label in ('nh4', 'do'):
            difference = errors[candidate][label] - errors[baseline][label]
            reported = summary['paired_day_differences'][candidate + '_minus_' + baseline][label]
            np.testing.assert_allclose(difference.mean(), reported['mean'], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(np.quantile(difference[indices].mean(1), [.025, .975]),
                reported['five_day_training_date_bootstrap95_ci'], rtol=1e-12, atol=1e-15)
    result = {'passed': True, 'validation_evaluated': False, 'test_evaluated': False, 'policy_training': False,
        'file_hashes_passed': True, 'strict_past_history_and_initialization_exact': True,
        'previous_public_air_matches_previous_action': True,
        'all_model_probabilities_exact_max_error': maximum,
        'all_and_new_event_metrics_reconstructed': True, 'date_intervals_reconstructed': True,
        'generated_training_transitions': len(x), 'held_training_transitions': int(rows.sum())}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
