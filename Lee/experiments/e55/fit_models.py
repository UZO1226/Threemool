"""Fit fixed one-step costs using only passively captured PPO transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from cost_model import fit_cost_models, features
from fast_cost import FastCost
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/code/darrow'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import frozen_digest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_capture(directory, data_path, smoke=False):
    files = sorted(directory.glob('update_*.npz'))
    assert len(files) == (1 if smoke else 66)
    training_days = set(map(str, pd.read_pickle(data_path)['days']['train']))
    collected = []
    hashes = {}
    for index, path in enumerate(files, 1):
        assert path.name == f'update_{index:03d}.npz'
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            row = {key: saved[key].copy() for key in saved.files}
        assert int(row.pop('update')) == index
        assert row['obs'].shape == row['next_obs'].shape == (760, 8, 11)
        assert set(np.unique(row['day'])).issubset(training_days)
        np.testing.assert_array_equal(row['nh4'], row['next_obs'][:, :, 0] > 4)
        np.testing.assert_array_equal(row['do'], row['next_obs'][:, :, 2:5].max(-1) > 2)
        assert np.isfinite(row['obs']).all() and np.isfinite(row['next_obs']).all()
        assert np.all(np.abs(row['action']) <= 1)
        np.testing.assert_array_equal(row['done'], row['within_episode'] == 94)
        assert row['done'].sum() == 64
        if collected:
            np.testing.assert_array_equal(row['episode'][0], collected[-1]['episode'][-1] + 1)
        np.testing.assert_array_equal(row['within_episode'][:, 0], np.tile(np.arange(95), 8))
        collected.append(row)
    joined = {key: np.concatenate([row[key] for row in collected]).reshape(
        -1, *collected[0][key].shape[2:]) for key in collected[0]}
    split = 4560 if smoke else 52 * 760 * 8
    fit = {key: value[:split] for key, value in joined.items()}
    held = {key: value[split:] for key, value in joined.items()}
    assert len(held['obs']) == (1520 if smoke else 14 * 760 * 8)
    return fit, held, hashes


def metrics(y, probability):
    y = np.asarray(y)
    return {'n': len(y), 'positive': int(y.sum()), 'prevalence': float(y.mean()),
            'brier': float(brier_score_loss(y, probability)),
            'average_precision': float(average_precision_score(y, probability)) if y.any() else None,
            'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            'recall_at_half': float(np.mean(probability[y == 1] >= .5)) if y.any() else None}


def verify(models, probes):
    fast = FastCost(models)
    probability = fast.probabilities(probes)
    expected = np.column_stack([model.predict_proba(probes)[:, 1] for model in models])
    np.testing.assert_allclose(probability, expected, rtol=1e-12, atol=1e-12)
    max_logit = 0.
    for index, model in enumerate(models):
        if hasattr(model, 'value'):
            continue
        reference = model.decision_function(probes)
        actual = fast.decision_function(probes)[:, index]
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
        max_logit = max(max_logit, float(np.max(np.abs(actual - reference))))
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    # Order audits use independently perturbed features, with identical other inputs.
    signs = {0: 1, 2: -1, 3: -1, 4: -1, 10: -1, 11: -1}
    worst = 0.
    for column, sign in signs.items():
        lower, upper = probes.copy(), probes.copy()
        if column == 11:
            lower[:, column], upper[:, column] = -1, 1
        else:
            lower[:, column] = 0
            upper[:, column] = np.maximum(probes[:, column], 1) * 1.5
        difference = (fast.probabilities(upper) - fast.probabilities(lower)) * np.array([sign, -sign])
        worst = max(worst, float(np.maximum(-difference, 0).max()))
    assert worst <= 1e-12
    return {'passed': True, 'probe_rows': len(probes),
            'max_fast_probability_difference': float(np.max(np.abs(probability - expected))),
            'max_fast_logit_difference': max_logit, 'max_coordinate_order_violation': worst}


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    progress = json.loads((args.run / 'progress.json').read_text())
    assert progress['phase'] == 'complete'
    assert progress['steps'] == (6080 if args.allow_smoke else 401280)
    config = json.loads((args.run / 'effective_config.json').read_text())
    assert config['method'] == 'ppolag_fcsrl' and not config['experiment_metadata']['test_evaluated']
    captured = Path(config['experiment_metadata']['passive_capture']['directory'])
    start = time.monotonic()
    fit, held, hashes = read_capture(captured, args.data, args.allow_smoke)
    models = fit_cost_models(fit['obs'], fit['action'], fit['nh4'], fit['do'])
    x = features(held['obs'], held['action'])
    probes = x[np.linspace(0, len(x) - 1, min(10000, len(x)), dtype=int)]
    checks = verify(models, probes)
    reports = {}
    for index, label in enumerate(('nh4', 'do')):
        probability = models[index].predict_proba(x)[:, 1]
        current = held['obs'][:, 0] > 4 if index == 0 else held['obs'][:, 2:5].max(-1) > 2
        reports[label] = {'held': metrics(held[label], probability),
                          'current_state_persistence': metrics(held[label], current.astype(float)),
                          'fit_prevalence_constant': metrics(held[label], np.full(len(x), fit[label].mean())),
                          'fit_positive': int(fit[label].sum()),
                          'constant_model': hasattr(models[index], 'value')}
    args.out.mkdir(parents=True)
    with (args.out / 'risk_models.pkl').open('wb') as file:
        pickle.dump(models, file, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.out / 'risk_models.pkl').open('rb') as file:
        restored = pickle.load(file)
    for original, saved in zip(models, restored):
        np.testing.assert_array_equal(original.predict_proba(probes), saved.predict_proba(probes))
    result = {'policy_seed': config['seed'], 'test_evaluated': False, 'smoke_only': args.allow_smoke,
              'base_training_steps': progress['steps'], 'extra_environment_steps': 0,
              'fit_transitions': len(fit['obs']), 'held_transitions': len(held['obs']),
              'fit_whole_updates': None if args.allow_smoke else 52,
              'held_whole_updates': None if args.allow_smoke else 14,
              'smoke_partition_only': 'First4560 versus last1520 of one update' if args.allow_smoke else None,
              'capture_file_sha256': hashes, 'checks': checks, 'label_diagnostics': reports,
              'model_sha256': digest(args.out / 'risk_models.pkl'),
              'source_sha256': {p.name: frozen_digest(p) for p in Path(__file__).parent.glob('*.py')},
              'data_sha256': digest(args.data), 'elapsed_seconds': time.monotonic() - start,
              'scope': 'Held on-policy training stream only; no validation fitting, tuning or refit. Diagnostic rows are not independent policy seeds.'}
    (args.out / 'fit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('COST_MODEL_FIT_PASS', json.dumps({'seed': config['seed'], 'checks': checks, 'held': reports}), flush=True)


if __name__ == '__main__':
    main()
