# Adapted from e71/audit.py at f9eac33; independently check guard before offset and disclose final-action risk trades.
# Adapted from Lee/experiments/e69/audit.py; e71 preserves e63 outside expanded-gain eligibility.
"""Independently replay the two shadow-price choices and convex action."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from guard_policy import ROOT, GuardPolicy as SignalPolicy, configuration, load_policy
from rate_io import load_base, load_rates
from policy_io import verify_sources
from rate_model import ConstantRate
from quantile_env import FrozenDynamics, evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'candidate', 'baseline', 'correction', 'projection', 'comparison', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--null-check', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    pair = json.loads((args.comparison / 'summary.json').read_text())
    assert pair['candidate_method'] == 'e73_component_guard_h4' and not pair['test_evaluated']
    assert pair['training_steps'] == 401280 and not pair['smoke_only']
    verify_sources(pair['source_sha256'], Path(__file__).parent)
    verify_sources(pair['shared_source_sha256'], ROOT)
    control_path = Path(__file__).with_name('guard_check.json')
    assert pair['control_certificate_sha256'] == digest(control_path)
    control = json.loads(control_path.read_text())
    assert control['passed']
    verify_sources(control['source_sha256'], ROOT)
    calibration = json.loads((args.projection / 'fit_summary.json').read_text())
    assert pair['projection_fit_summary_sha256'] == digest(args.projection / 'fit_summary.json')
    assert calibration['passed'] and calibration['calibration']['absolute_mean_action_error'] <= 1e-10
    base, config, metrics, hashes = load_base(args.candidate, args.freeze, args.baseline)
    models, fit = load_rates(args.correction, config, 4, False)
    assert hashes == pair['base_checkpoint_hashes'] == calibration['base_checkpoint_hashes']
    assert fit['model_sha256'] == pair['correction_model_sha256'] == calibration['rate_model_sha256']
    path = args.comparison / 'controller_config.json'
    assert digest(path) == pair['controller_config_sha256'] == calibration['controller_config_sha256']
    policy = load_policy(base, models, path)
    assert policy.config == calibration['controller_config']
    dynamics = FrozenDynamics(args.freeze)
    offsets = [('native', 0.)]
    if pair['candidate_matched'] is not None:
        offsets.append(('matched', pair['matching']['offset']))
    replays = {}
    for phase, offset in offsets:
        totals = {'rows': 0, 'volume': 0., 'wrapper_component_increase': np.zeros(2, dtype=int),
                  'executed_component_increase_vs_same_state_matched_e63': np.zeros(2, dtype=int),
                  'executed_component_delta_sum_vs_same_state_matched_e63': np.zeros(2)}
        def act(obs):
            preserved = obs.copy()
            original = np.asarray(base.act_det(obs)).reshape(-1)
            candidates = np.clip(original[:, None] + [0, -.125, .125, -.25, .25, -.375, .375, -.5, .5], -1, 1)
            x = np.column_stack((np.repeat(obs, 9, axis=0), candidates.reshape(-1)))
            components = np.clip(np.column_stack([model.predict(x) for model in models]), 0, 1).reshape(-1, 9, 2)
            raw_risk = components.sum(-1)
            reference_index = raw_risk[:, :5].argmin(-1)
            reference_action = candidates[np.arange(len(obs)), reference_index]
            reference_components = components[np.arange(len(obs)), reference_index]
            admissible = np.all(components <= reference_components[:, None, :] + 1e-12, axis=-1)
            candidates = np.where(admissible, candidates, reference_action[:, None])
            risk = np.where(admissible, raw_risk, reference_components.sum(-1)[:, None])
            low_cost = risk + policy.config['lambda_low'] * (candidates - candidates[:, :1])
            high_cost = risk + policy.config['lambda_high'] * (candidates - candidates[:, :1])
            low = candidates[np.arange(len(obs)), low_cost.argmin(-1)]
            high = candidates[np.arange(len(obs)), high_cost.argmin(-1)]
            weight = policy.config['low_policy_weight']
            expected = low if policy.config['lambda_low'] == policy.config['lambda_high'] else weight * low + (1 - weight) * high
            actual = policy.act_det(obs)
            signal = (risk[:, :5].min(-1) - risk.min(-1) > 1e-12) & (not policy.config['reference_only'])
            reference = candidates[np.arange(len(obs)), risk[:, :5].argmin(-1)]
            expected = np.where(signal, expected, reference)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(obs, preserved)
            assert np.max(np.abs(actual - original)) <= .5 + 1e-12
            assert np.isfinite(actual).all() and np.all(np.abs(actual) <= 1)
            wrapper_x = np.column_stack((obs, actual))
            wrapper_components = np.clip(np.column_stack([model.predict(wrapper_x) for model in models]), 0, 1)
            wrapper_excess = wrapper_components - reference_components
            totals['wrapper_component_increase'] += (wrapper_excess > 1e-12).sum(0)
            assert np.all(wrapper_excess <= 1e-12)
            executed = np.clip(actual + offset, -1, 1)
            reference_executed = np.clip(reference_action + pair['baseline_fixed_offset'], -1, 1)
            final_x = np.column_stack((obs, executed))
            reference_x = np.column_stack((obs, reference_executed))
            final_components = np.clip(np.column_stack([model.predict(final_x) for model in models]), 0, 1)
            primary_components = np.clip(np.column_stack([model.predict(reference_x) for model in models]), 0, 1)
            final_delta = final_components - primary_components
            totals['executed_component_increase_vs_same_state_matched_e63'] += (final_delta > 1e-12).sum(0)
            totals['executed_component_delta_sum_vs_same_state_matched_e63'] += final_delta.sum(0)
            totals['volume'] += float(np.sum((executed + 1) / 2 * 9000 * .25))
            totals['rows'] += len(obs)
            return actual
        summary, rows = evaluate(args.data, dynamics, act_batch=act, air_offset=offset)
        assert json.loads((args.comparison / f'candidate_{phase}.json').read_text()) == {'summary': summary, 'rows': rows}
        assert totals['rows'] == 152000
        np.testing.assert_allclose(totals['volume'] / 1600, summary['air_m3'], rtol=1e-12, atol=1e-8)
        replays[phase] = {'independent_sklearn_actions_exact': True, 'trajectory_replay_exact': True,
            'executed_air_accounting_exact': True, 'action_rows': totals['rows'], 'offset': offset,
            'wrapper_component_guard_passed': True,
            'wrapper_component_increase_rows': totals['wrapper_component_increase'].tolist(),
            'executed_component_increase_rows_vs_same_state_matched_e63': totals['executed_component_increase_vs_same_state_matched_e63'].tolist(),
            'mean_executed_predicted_component_delta_vs_same_state_matched_e63': (totals['executed_component_delta_sum_vs_same_state_matched_e63'] / totals['rows']).tolist(),
            'guard_scope': 'Pre-offset wrapper only; post-offset predicted changes are descriptive and not constrained.'}
        print('BUDGET_INDEPENDENT_REPLAY', config['seed'], phase, flush=True)
    if args.null_check:
        null = SignalPolicy(base, [ConstantRate(.3), ConstantRate(.8)], policy.config)
        summary, _ = evaluate(args.data, dynamics, act_batch=null.act_det)
        for key, value in metrics['val_final'].items():
            np.testing.assert_allclose(summary[key], value, rtol=1e-10, atol=1e-10)
        reference_config = configuration(policy.config['lambda_low'], policy.config['lambda_high'],
                                         policy.config['low_policy_weight'], enabled=False)
        unconstrained = SignalPolicy(base, models, reference_config)
        summary, rows = evaluate(args.data, dynamics, act_batch=unconstrained.act_det)
        previous = ROOT.parent / f'results/e63_study/comparisons/seed_{config["seed"]}/candidate_native.json'
        assert json.loads(previous.read_text()) == {'summary': summary, 'rows': rows}
    result = {'passed': True, 'policy_seed': config['seed'], 'replays': replays,
        'flat_risk_fitted_parameters_null_full_native_exact': args.null_check,
        'reference_only_e63_full_native_exact': args.null_check,
        'comparison_summary_sha256': digest(args.comparison / 'summary.json'),
        'audit_source_sha256': digest(Path(__file__)), 'test_evaluated': False}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('BUDGET_AUDIT_COMPLETE', config['seed'], flush=True)


if __name__ == '__main__':
    main()
