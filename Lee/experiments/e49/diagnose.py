"""Observed-state next-event diagnostics, separate from autonomous simulation."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics
from quantile_env import FrozenDynamics
from quantile_dynamics import arrays, paths, rollout, features
from event_timing import onset_counts


def observed_arrays(table, days):
    truth, exo, zone = arrays(table, days)
    assert truth.shape == (len(days), 96, 5)
    previous = np.concatenate([zone[:, :1], zone[:, :-2]], axis=1)
    x = features(truth[:, :-1].reshape(-1, 5), exo[:, :-1].reshape(-1, 9),
                 zone[:, :-1].reshape(-1, 5), previous.reshape(-1, 5))
    target = np.stack([truth[:, 1:, 0] > 4, truth[:, 1:, 2:].max(-1) > 2], -1)
    current = np.stack([truth[:, :-1, 0] > 4, truth[:, :-1, 2:].max(-1) > 2], -1)
    return x, target, current, (truth, exo, zone)


def classify_metrics(probability, target):
    probability = np.asarray(probability, float)
    target = np.asarray(target, bool)
    positive = probability >= .5
    tp = int((positive & target).sum())
    fp = int((positive & ~target).sum())
    fn = int((~positive & target).sum())
    bins = []
    for i in range(10):
        mask = (probability >= i / 10) & ((probability < (i + 1) / 10) if i < 9 else (probability <= 1))
        bins.append({'count': int(mask.sum()),
                     'mean_probability': float(probability[mask].mean()) if mask.any() else None,
                     'observed_frequency': float(target[mask].mean()) if mask.any() else None})
    assert sum(row['count'] for row in bins) == target.size
    return {'count': int(target.size), 'positives': int(target.sum()),
            'prevalence': float(target.mean()), 'mean_probability': float(probability.mean()),
            'threshold_frequency': float(positive.mean()),
            'brier': float(((probability - target) ** 2).mean()),
            'average_precision': float(average_precision_score(target, probability)) if target.any() else None,
            'roc_auc': float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
            'precision_at_half': tp / (tp + fp) if tp + fp else None,
            'recall_at_half': tp / (tp + fn) if tp + fn else None,
            'reliability_bins': bins}


def frozen_probabilities(dyn, batch, days):
    truth, exo, zone = batch
    n = len(days)
    noise = paths(dyn.bank, days, 95)
    native = rollout(dyn.model, batch, noise, dyn.alpha)
    zone = zone.copy()
    zone *= np.minimum(1, 9000 / np.maximum(zone.sum(-1, keepdims=True), 1))
    members = np.repeat(np.arange(5), 8 * n)
    reset = []
    for t in range(95):
        state = np.tile(truth[:, t], (40, 1))
        reset.append(dyn.step_with_noise(state, np.tile(exo[:, t], (40, 1)),
                     np.tile(zone[:, t], (40, 1)), np.tile(zone[:, max(0, t - 1)], (40, 1)),
                     members, noise[:, t]))
    reset = np.stack(reset, 1).reshape(5, 8, n, 95, 5)
    def probabilities(trajectory):
        return np.stack([(trajectory[..., 0] > 4).mean((0, 1)),
                         (trajectory[..., 2:].max(-1) > 2).mean((0, 1))], -1)
    return {'e40_free': probabilities(native), 'e40_observed': probabilities(reset)}, native


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw = pd.read_pickle(args.data)
    reference = json.loads(args.reference_summary.read_text())
    fit, calibration, validation = reference['fit_days'], reference['calibration_days'], raw['days']['val']
    assert len(fit) == 158 and len(calibration) == 30 and len(validation) == 40
    assert set(fit).isdisjoint(calibration) and set(fit + calibration).isdisjoint(validation)
    assert set(fit + calibration + validation).isdisjoint(raw['days']['test'])
    table = raw['data'].loc[:validation[-1]].copy()
    assert table.index.max() < pd.Timestamp(validation[-1]) + pd.Timedelta(days=1)
    x, target, _, _ = observed_arrays(table, fit)
    models, subsets = {}, {}
    columns = {'full': np.arange(24), 'no_state': np.arange(5, 24)}
    for name, cols in columns.items():
        models[name] = []
        for member in range(5):
            selected = np.random.RandomState(member).choice(len(fit), int(.8 * len(fit)), replace=False)
            subsets[str(member)] = [fit[i] for i in selected]
            rows = (selected[:, None] * 95 + np.arange(95)).reshape(-1)
            member_models = []
            for label in range(2):
                y = target.reshape(-1, 2)[rows, label]
                assert len(np.unique(y)) == 2
                model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                    max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1,
                    early_stopping=False, random_state=100 * member + label)
                model.fit(x[rows][:, cols], y)
                member_models.append(model)
                print('FIT', name, member, label, 'positives', int(y.sum()), flush=True)
            models[name].append(member_models)
    with (args.out / 'models.pkl').open('wb') as stream:
        pickle.dump({'models': models, 'columns': columns}, stream)
    dyn = FrozenDynamics(args.freeze)
    result = {'test_evaluated': False, 'policy_training': False, 'adoption_claim': False,
              'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                   'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'sklearn_version': sklearn.__version__, 'fit_days': fit, 'calibration_days': calibration,
              'validation_days': validation, 'model_date_subsets': subsets,
              'diagnostic': 'Observed-state next-step prediction is not free-running simulation or causal air response.',
              'fit_counts': {label: {'positives': int(target[..., j].sum()), 'total': int(target[..., j].size)}
                             for j, label in enumerate(('nh4', 'do'))},
              'splits': {}}
    errors = {}
    for split, days in [('calibration', calibration), ('validation', validation)]:
        xx, yy, current, batch = observed_arrays(table, days)
        predictions, free = frozen_probabilities(dyn, batch, days)
        predictions['persistence'] = current.astype(float)
        for name, cols in columns.items():
            member_predictions = np.stack([np.stack([model.predict_proba(xx[:, cols])[:, 1]
                for model in member_models], -1).reshape(len(days), 95, 2) for member_models in models[name]])
            predictions[name] = member_predictions.mean(0)
        result['splits'][split] = {}
        for name, probabilities in predictions.items():
            assert probabilities.shape == yy.shape and np.isfinite(probabilities).all()
            rows = {}
            for j, label in enumerate(('nh4', 'do')):
                probability, target_j = probabilities[..., j], yy[..., j]
                eligible = ~current[..., j]
                rows[label] = {'all_steps': classify_metrics(probability.ravel(), target_j.ravel()),
                               'currently_nonviolating': classify_metrics(probability[eligible], target_j[eligible])}
                onsets = {}
                for tolerance in (0, 1, 4):
                    counts = np.sum([onset_counts(y, p >= .5, initial, tolerance)
                        for y, p, initial in zip(target_j, probability, current[:, 0, j])], axis=0)
                    actual, predicted, matched, delay = counts
                    onsets[str(tolerance)] = {'actual_onsets': int(actual), 'predicted_onsets': int(predicted),
                        'matched': int(matched), 'recall': float(matched / actual) if actual else None,
                        'precision': float(matched / predicted) if predicted else None,
                        'mean_absolute_delay_steps': float(delay / matched) if matched else None}
                rows[label]['onset_tolerance_steps'] = onsets
                if split == 'validation':
                    errors.setdefault(name, {})[label] = ((probability - target_j) ** 2).mean(1)
            result['splits'][split][name] = rows
        free_metrics, _ = event_metrics(free, batch[0])
        if split == 'validation':
            prior = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
            for label in ('nh4', 'do'):
                for key in ('brier', 'average_precision', 'roc_auc'):
                    np.testing.assert_allclose(free_metrics[label][key], prior['metrics']['e40'][label][key], rtol=1e-12)
        result['splits'][split]['e40_free_onsets'] = free_metrics
        result['splits'][split]['recorded_air_steps_above_cap'] = int((batch[2][:, :-1].sum(-1) > 9000).sum())
        np.savez_compressed(args.out / f'{split}_probabilities.npz', **predictions)
        print('EVALUATED', split, flush=True)
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, len(validation), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(validation)).reshape(10000, -1)[:, :len(validation)]
    result['paired_date_brier_differences'] = {}
    for candidate, baseline in [('full', 'no_state'), ('full', 'persistence'),
                                 ('full', 'e40_observed'), ('e40_observed', 'e40_free')]:
        comparison = {}
        for label in ('nh4', 'do'):
            diff = errors[candidate][label] - errors[baseline][label]
            comparison[label] = {'mean': float(diff.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(diff[indices].mean(1), [.025, .975]).tolist()}
        result['paired_date_brier_differences'][candidate + '_minus_' + baseline] = comparison
    result['model_sha256'] = hashlib.sha256((args.out / 'models.pkl').read_bytes()).hexdigest()
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['paired_date_brier_differences']), flush=True)


if __name__ == '__main__':
    main()
