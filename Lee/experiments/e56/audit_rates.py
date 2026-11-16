"""Independently audit held rate errors or actual fixed-controller choices."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rate_io import load_base, load_rates, FrozenDynamics
from rate_model import ConstantRate, RatePolicy
from streams import read_stream
from quantile_env import evaluate


def held_audit(args, models, summary):
    fit, held, provenance = read_stream(args.run, args.data, args.horizon, args.allow_smoke)
    for key in ('capture_file_sha256', 'fit_updates', 'held_updates', 'policy_seed', 'data_sha256'):
        assert provenance[key] == summary[key]
    prediction = np.clip(np.column_stack([model.predict(held['x']) for model in models]), 0, 1)
    days = np.unique(held['day'])
    rng = np.random.default_rng(5602)
    bootstrap = rng.integers(0, len(days), size=(10000, len(days)))
    counts = np.array([(held['day'] == day).sum() for day in days])
    reports = {}
    for index, label in enumerate(('nh4', 'do')):
        y, pred = held['target'][:, index], prediction[:, index]
        stored = summary['label_diagnostics'][label]
        np.testing.assert_allclose(np.mean(np.square(pred - y)), stored['held']['mse'], atol=1e-15, rtol=1e-12)
        np.testing.assert_allclose(np.mean(np.abs(pred - y)), stored['held']['mae'], atol=1e-15, rtol=1e-12)
        np.testing.assert_allclose(np.mean(pred - y), stored['held']['mean_bias'], atol=1e-15, rtol=1e-12)
        current = held['x'][:, 0] > 4 if index == 0 else held['x'][:, 2:5].max(-1) > 2
        controls = {'current_state_persistence': current.astype(float),
                    'fit_mean_constant': np.full(len(y), fit['target'][:, index].mean())}
        contrasts = {}
        for name, control in controls.items():
            np.testing.assert_allclose(np.mean(np.square(control - y)), stored[name]['mse'], atol=1e-15, rtol=1e-12)
            difference = np.square(pred - y) - np.square(control - y)
            sums = np.array([difference[held['day'] == day].sum() for day in days])
            distribution = sums[bootstrap].sum(1) / counts[bootstrap].sum(1)
            contrasts[name] = {'rate_mse_difference': float(difference.mean()),
                               'paired_date_bootstrap95_ci': np.quantile(distribution, [.025, .975]).tolist()}
        safe = ~current
        reports[label] = {'rate_mse': float(np.mean(np.square(pred - y))),
                          'target_mean': float(y.mean()), 'prediction_mean': float(pred.mean()),
                          'safe_current_rows': int(safe.sum()),
                          'safe_current_future_nonzero_windows': int(np.count_nonzero(y[safe] > 0)),
                          'safe_current_rate_mse': float(np.mean(np.square(pred[safe] - y[safe]))) if safe.any() else None,
                          'controls': contrasts}
    return {'held_metrics_reconstructed': True, 'held_rows': len(held['x']),
            'held_training_dates': len(days), 'labels': reports,
            'uncertainty_scope': '10000 held training-date resamples conditional on this fitted policy/model; not policy-seed, causal-action or plant-generalization uncertainty.'}


def action_audit(args, base, models):
    assert args.comparison is not None
    policy = RatePolicy(base, models)
    totals = {'n': 0, 'changed': 0, 'clipped': 0, 'risk_change': 0., 'squared_adjustment': 0.}
    counts = np.zeros(5, dtype=int)
    def act(obs):
        original = np.asarray(base.act_det(obs)).reshape(-1)
        raw = original[:, None] + np.array([0, -.125, .125, -.25, .25])
        candidates = np.clip(raw, -1, 1)
        x = np.column_stack((np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
        # NO3 stays in the actual current public input; no future state enters.
        np.testing.assert_array_equal(x[:, 1], np.repeat(obs[:, 1], 5))
        rates = np.clip(np.column_stack([model.predict(x) for model in models]), 0, 1)
        risk = rates.sum(-1).reshape(-1, 5)
        score = risk + np.square(candidates - original[:, None])
        selected = np.argmin(score, axis=1)
        expected = candidates[np.arange(len(obs)), selected]
        actual = policy.act_det(obs)
        np.testing.assert_array_equal(actual, expected)
        counts[:] += np.bincount(selected, minlength=5)
        totals['n'] += len(obs)
        totals['changed'] += int(np.count_nonzero(actual != original))
        totals['clipped'] += int(np.count_nonzero(raw != candidates))
        totals['risk_change'] += float((risk[np.arange(len(obs)), selected] - risk[:, 0]).sum())
        totals['squared_adjustment'] += float(np.square(actual - original).sum())
        return actual
    dynamics = FrozenDynamics(args.freeze)
    summary, rows = evaluate(args.data, dynamics, act_batch=act)
    saved = json.loads((args.comparison / 'candidate_native.json').read_text())
    assert summary == saved['summary'] and rows == saved['rows']
    result = {'independent_sklearn_choices_exact': True, 'native_replay_exact': True,
              'action_rows': totals['n'], 'candidate_choice_counts': counts.tolist(),
              'changed_action_fraction': totals['changed'] / totals['n'],
              'candidate_clipped_fraction': totals['clipped'] / (5 * totals['n']),
              'mean_predicted_rate_change': totals['risk_change'] / totals['n'],
              'mean_squared_adjustment': totals['squared_adjustment'] / totals['n']}
    if args.null_check:
        null = RatePolicy(base, [ConstantRate(.3), ConstantRate(.8)])
        null_summary, null_rows = evaluate(args.data, dynamics, act_batch=null.act_det)
        original = json.loads((args.comparison / 'baseline_native.json').read_text())
        assert null_summary == original['summary'] and null_rows == original['rows']
        result['flat_rate_null_full_day_exact'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'baseline', 'freeze', 'data', 'correction', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--horizon', type=int, choices=(1, 4), required=True)
    parser.add_argument('--phase', choices=('held', 'actions', 'both'), default='held')
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--null-check', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    base, config, _, hashes = load_base(args.run, args.freeze, args.baseline)
    models, summary = load_rates(args.correction, config, args.horizon, args.allow_smoke)
    assert hashes == summary['base_checkpoint_hashes']
    result = {'passed': True, 'policy_seed': config['seed'], 'horizon': args.horizon,
              'phase': args.phase, 'smoke_only': args.allow_smoke, 'test_evaluated': False,
              'scope': 'Fixed rate controller, unchanged base policy and held on-policy captured data. Predicted rate improvements are not measured water-quality improvements.'}
    if args.phase in ('held', 'both'):
        result['held'] = held_audit(args, models, summary)
    if args.phase in ('actions', 'both'):
        result['actions'] = action_audit(args, base, models)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('RATE_AUDIT_PASS', config['seed'], args.horizon, args.phase, flush=True)


if __name__ == '__main__':
    main()
