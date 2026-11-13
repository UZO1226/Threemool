"""Probe strictly past public observations on training-only frozen episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import FrozenDynamics, FrozenQuantileEnv

LAG_COLUMNS = [0, 1, 2, 3, 4, 10]


def metrics(probability, target):
    if len(target) == 0:
        return {'count': 0, 'positives': 0, 'brier': None, 'average_precision': None, 'roc_auc': None}
    positive = probability >= .5
    tp = int((positive & target).sum())
    fp = int((positive & ~target).sum())
    fn = int((~positive & target).sum())
    return {'count': len(target), 'positives': int(target.sum()),
            'prevalence': float(target.mean()), 'brier': float(np.mean((probability - target) ** 2)),
            'average_precision': float(average_precision_score(target, probability)) if target.any() else None,
            'roc_auc': float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
            'precision_at_half': tp / (tp + fp) if tp + fp else None,
            'recall_at_half': tp / (tp + fn) if tp + fn else None}


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw = pd.read_pickle(args.data)
    ref = json.loads(args.reference_summary.read_text())
    fit, held = ref['fit_days'], ref['calibration_days']
    assert len(fit) == 158 and len(held) == 30 and fit + held == raw['days']['train']
    assert set(fit + held).isdisjoint(raw['days']['val'] + raw['days']['test'])
    dynamics = FrozenDynamics(args.freeze)
    expected = next(value for name, value in dynamics.manifest['frozen_files_sha256'].items()
                    if name.endswith('darrow_15min.pkl'))
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == expected
    cases = [(day, member) for member in range(5) for day in fit + held]
    env = FrozenQuantileEnv(args.data, dynamics, split='train', n_env=len(cases), seed=5301)
    env.fixed, env.fixed_draws = cases, [0] * len(cases)
    obs = env.reset()
    lag = obs[:, LAG_COLUMNS].copy()
    rng = np.random.default_rng(5301)
    inputs, labels, current = [], [], []
    for t in range(95):
        action = rng.uniform(-1, 1, len(cases))
        if t == 0:
            np.testing.assert_array_equal(lag, obs[:, LAG_COLUMNS])
        else:
            np.testing.assert_array_equal(lag, inputs[-1][:, LAG_COLUMNS])
        inputs.append(np.c_[obs, lag, action].astype(np.float32))
        current.append(np.c_[obs[:, 0] > 4, obs[:, 2:5].max(-1) > 2])
        following, reward, cost, truncated, terminal, info = env.step(action)
        np.testing.assert_allclose(info['air'], (action + 1) / 2 * 9000, rtol=1e-12)
        np.testing.assert_allclose(reward, -info['air'] / 9000, rtol=1e-12)
        np.testing.assert_array_equal(cost, info['v_nh4'] + info['v_do'])
        np.testing.assert_array_equal(info['done_day'], np.array([day for day, _ in cases]))
        assert truncated.all() if t == 94 else not truncated.any()
        labels.append(np.c_[info['v_nh4'], info['v_do']].astype(bool))
        lag, obs = obs[:, LAG_COLUMNS].copy(), following
        if (t + 1) % 20 == 0:
            print('GENERATED_STEP', t + 1, flush=True)
    x, target, current = np.concatenate(inputs), np.concatenate(labels), np.concatenate(current)
    case_days = np.array([day for day, _ in cases])
    row_days = np.tile(case_days, 95)
    assert np.isfinite(x).all() and x.shape == (89300, 18)
    repeated = x.copy()
    repeated[:, 11:17] = repeated[:, LAG_COLUMNS]
    np.savez_compressed(args.out / 'generated.npz', x=x, target=target, current=current, row_days=row_days)
    splits = {'current': (x, np.r_[np.arange(11), 17]),
              'history': (x, np.arange(18)), 'repeated_current': (repeated, np.arange(18))}
    held_rows = np.isin(row_days, held)
    models, predictions = {}, {}
    result = {'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'validation_evaluated': False, 'test_evaluated': False, 'policy_training': False,
              'probe_fit_seeds': list(range(5)), 'environment_members': list(range(5)), 'policy_seeds': [],
              'action_rng_seed': 5301, 'noise_draws_per_case': 1, 'fit_days': fit, 'held_training_days': held,
              'data_sha256': expected, 'freeze_sha256': hashlib.sha256(args.freeze.read_bytes()).hexdigest(),
              'training_transitions_generated': len(x), 'held_training_transitions': int(held_rows.sum()),
              'scope': 'Feature diagnostic within a frozen simulated law, not independent plant confirmation.',
              'checks': {'strict_past_lags_exact': True, 'initial_lag_repeats_current': True,
                         'no_cross_reset_history': True, 'terminal_air_reward_cost_exact': True,
                         'no_hidden_model_or_noise_features': True}, 'metrics': {}, 'paired_day_differences': {}}
    daily_errors = {}
    for name, (features, columns) in splits.items():
        ensemble, fitted = [], []
        for seed in range(5):
            dates = [fit[i] for i in np.random.RandomState(seed).choice(158, int(.8 * 158), replace=False)]
            rows = np.isin(row_days, dates)
            member, probability = [], []
            for j in range(2):
                y = target[rows, j]
                assert len(np.unique(y)) == 2
                model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                    min_samples_leaf=40, l2_regularization=1, early_stopping=False, random_state=100 * seed + j)
                model.fit(features[rows][:, columns], y)
                member.append(model)
                probability.append(model.predict_proba(features[held_rows][:, columns])[:, 1])
            fitted.append(member)
            ensemble.append(np.stack(probability, -1))
            print('FIT_PROBE', name, seed, flush=True)
        models[name] = {'models': fitted, 'columns': columns}
        probability = np.mean(ensemble, axis=0)
        predictions[name] = probability
        result['metrics'][name], daily_errors[name] = {}, {}
        held_days_by_row = row_days[held_rows]
        for j, label in enumerate(('nh4', 'do')):
            y, p = target[held_rows, j], probability[:, j]
            eligible = ~current[held_rows, j]
            result['metrics'][name][label] = {'all_steps': metrics(p, y),
                'currently_nonviolating': metrics(p[eligible], y[eligible]),
                'member_brier': [float(np.mean((values[:, j] - y) ** 2)) for values in ensemble]}
            daily_errors[name][label] = np.array([np.mean((p[held_days_by_row == day] - y[held_days_by_row == day]) ** 2)
                                                for day in held])
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, 30, (10000, 6))
    indices = ((starts[:, :, None] + np.arange(5)) % 30).reshape(10000, 30)
    for candidate, baseline in [('history', 'current'), ('history', 'repeated_current'), ('repeated_current', 'current')]:
        comparison = {}
        for label in ('nh4', 'do'):
            difference = daily_errors[candidate][label] - daily_errors[baseline][label]
            comparison[label] = {'mean': float(difference.mean()),
                'five_day_training_date_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
        result['paired_day_differences'][candidate + '_minus_' + baseline] = comparison
    with (args.out / 'models.pkl').open('wb') as stream:
        pickle.dump(models, stream)
    np.savez_compressed(args.out / 'held_probabilities.npz', **predictions)
    result['model_sha256'] = hashlib.sha256((args.out / 'models.pkl').read_bytes()).hexdigest()
    result['generated_sha256'] = hashlib.sha256((args.out / 'generated.npz').read_bytes()).hexdigest()
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['paired_day_differences']), flush=True)


if __name__ == '__main__':
    main()
