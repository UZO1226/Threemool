# Adapted from e74/fit_anchor.py at0e35636, adding the preregistered e78 upper guard.
# Adapted from e73/fit_guard.py at e2bd4f3; calibration and geometry use authenticated matched e63.
# Adapted from e71/fit_signal.py at f9eac33; hard component exclusions precede identical mean calibration.
"""Calibrate only fitting-state mean action, then diagnose held choices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from shield_policy import ROOT, ShieldPolicy as SignalPolicy, AnchorPolicy, anchor_configuration, calibrate, configuration as shield_configuration, load_policy, project, public_features, eligible
from anchor_policy import TrustPolicy
sys.path.insert(0, str(ROOT / 'Lee/experiments/e73'))
from guard_policy import GuardPolicy as PreviousPolicy
from rate_io import load_base, load_rates
from rate_model import slow_rates
from streams import read_stream
from policy_io import verify_sources



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'certificate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    control = json.loads((ROOT / 'Lee/experiments/e74/anchor_check.json').read_text())
    forecast_path = args.workspace.resolve() / 'results/e78_do_upper'
    forecast_summary = json.loads((forecast_path / 'summary.json').read_text())
    forecast_audit = json.loads((forecast_path / 'audit.json').read_text())
    assert forecast_summary['all_models_admitted'] and forecast_audit['passed'] and forecast_audit['all_models_admitted']
    assert sha(forecast_path / 'summary.json') == forecast_audit['summary_sha256']
    verify_sources(forecast_summary['source_sha256'], ROOT)
    assert control['passed'] and control['new_environment_queries'] == 0
    verify_sources(control['source_sha256'], ROOT)
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    certificate = json.loads(args.certificate.read_text())
    assert certificate['passed'] and certificate['policy_seeds'] == list(range(5))
    assert sha(data) == certificate['data_sha256']
    prior = json.loads((ROOT / 'Lee/experiments/e69/preflight.json').read_text())
    verify_sources({'budget_policy.py': prior['source_sha256']['budget_policy.py']}, ROOT / 'Lee/experiments/e69')
    pinned = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    args.out.mkdir(parents=True)
    records = []
    for seed in range(5):
        assert all(sha(Path(__file__).parent / name) == digest for name, digest in pinned.items())
        base_path = workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}'
        original_path = workspace / f'results/e42_baselines/new_ppolag_fcsrl/seed_{seed}'
        base, config, _, hashes = load_base(base_path, freeze, original_path)
        models, fit = load_rates(workspace / f'results/e56_study/models/h4/seed_{seed}', config, 4, False)
        reference = certificate['records'][seed]
        assert hashes == fit['base_checkpoint_hashes'] == reference['base_checkpoint_hashes']
        assert fit['model_sha256'] == reference['rate_model_sha256']
        fitting, held, provenance = read_stream(base_path, data, 4, False)
        obs_fit, obs_held = fitting['x'][:, :11], held['x'][:, :11]
        assert len(obs_fit) == 316160 and len(obs_held) == 85120
        reference_offset = reference['fixed_e63_offset_after_wrapper']
        def configuration(low, high, weight, offset, enabled=True):
            return shield_configuration(low, high, weight, offset, seed, forecast_summary['records'][seed], enabled)
        zero = SignalPolicy(base, models, configuration(0, 0, 1, reference_offset))
        candidates, risks, targets = [], [], []
        for start in range(0, len(obs_fit), 2048):
            c, r = zero.geometry(obs_fit[start:start + 2048])
            old_action = c[:, 0]
            candidates.append(c)
            risks.append(r)
            targets.append(old_action)
        candidates, risks = np.concatenate(candidates), np.concatenate(risks)
        old_actions = np.concatenate(targets)
        target = float(old_actions.mean())
        active = eligible(risks)
        if active.any():
            fitted, calibration = calibrate(candidates[active], risks[active], float(old_actions[active].mean()))
            projected_config = configuration(fitted['lambda_low'], fitted['lambda_high'], fitted['low_policy_weight'], reference_offset)
        else:
            projected_config = configuration(0, 0, 1, reference_offset, enabled=False)
            calibration = {'absolute_mean_action_error': 0., 'trace': [], 'scope': 'No eligible fitting rows.'}
        active_count = int(active.sum())
        global_error = abs(float(project(candidates, risks, projected_config).mean()) - target)
        assert global_error <= 1e-10
        before_mean = float(project(candidates, risks, zero.config).mean())
        del candidates, risks, targets, old_actions, active
        folder = args.out / f'seed_{seed}'
        folder.mkdir()
        policy = SignalPolicy(base, models, projected_config)
        policy.save_config(folder / 'controller_config.json')
        restored = load_policy(base, models, folder / 'controller_config.json')
        previous_config = json.loads((workspace / f'results/e74_anchor_fit/seed_{seed}/controller_config.json').read_text())
        previous_policy = AnchorPolicy(base, models, previous_config)
        disabled_config = dict(previous_config, do_shield=dict(projected_config['do_shield'], enabled=False))
        disabled_policy = SignalPolicy(base, models, disabled_config)
        old = TrustPolicy(base, models, 0)
        old_projection = json.loads((workspace / f'results/e73_guard_fit/seed_{seed}/fit_summary.json').read_text())
        assert old_projection['base_checkpoint_hashes'] == hashes
        legacy = PreviousPolicy(base, models, old_projection['controller_config'])
        changed = eligible_held = legacy_changed = legacy_without_signal = worse_risk = 0
        risk_delta_sum = 0.
        largest_component_increase = np.zeros(2)
        held_original, held_new = [], []
        for start in range(0, len(obs_held), 2048):
            obs = obs_held[start:start + 2048]
            np.testing.assert_array_equal(disabled_policy.act_det(obs), previous_policy.act_det(obs))
            unchanged = obs.copy()
            c, r = policy.geometry(obs)
            action = project(c, r, projected_config)
            legacy_action = legacy.act_det(obs)
            previous_native = old.act_det(obs)
            previous = np.clip(previous_native + reference_offset, -1, 1)
            np.testing.assert_array_equal(previous, c[:, 0])
            mask = eligible(r) & (not projected_config['reference_only'])
            np.testing.assert_array_equal(action[~mask], previous[~mask])
            eligible_held += int(mask.sum())
            legacy_changed += int(np.count_nonzero(legacy_action != previous_native))
            legacy_without_signal += int(np.count_nonzero((legacy_action != previous_native) & ~eligible(r)))
            actual_upper = policy.upper(obs, action, original=True)
            reference_upper = policy.upper(obs, previous, original=True)
            assert np.all(actual_upper <= np.maximum(2., reference_upper) + 1e-12)
            actual_components = policy.risk.predict(public_features(obs, action))
            reference_components = policy.risk.predict(public_features(obs, previous))
            excess = actual_components - reference_components
            assert np.all(excess <= 1e-12)
            largest_component_increase = np.maximum(largest_component_increase, excess.max(0))
            actual_risk = actual_components.sum(-1)
            delta_risk = actual_risk - r[:, 0]
            worse_risk += int(np.count_nonzero(delta_risk > 1e-12))
            risk_delta_sum += float(delta_risk.sum())
            assert np.all(np.abs(action - previous) <= .5 + 1e-12)
            assert np.isfinite(action).all() and np.all(np.abs(action) <= 1)
            np.testing.assert_array_equal(obs, unchanged)
            changed += int(np.count_nonzero(action != previous))
            held_original.extend(previous.tolist())
            held_new.extend(action.tolist())
        assert legacy_changed == old_projection['held_changed_action_rows']
        subset = obs_held[np.linspace(0, len(obs_held) - 1, 1024, dtype=int)]
        original = np.asarray(base.act_det(subset)).reshape(-1)
        old_candidates = np.clip(original[:, None] + [0, -.125, .125, -.25, .25], -1, 1)
        old_x = public_features(np.repeat(subset, 5, axis=0), old_candidates.reshape(-1))
        old_risk = slow_rates(models, old_x).sum(-1).reshape(-1, 5)
        reference_action = np.clip(old_candidates[np.arange(len(subset)), old_risk.argmin(-1)] + reference_offset, -1, 1)
        c = np.clip(reference_action[:, None] + [0, -.125, .125, -.25, .25, -.375, .375, -.5, .5], -1, 1)
        x = public_features(np.repeat(subset, 9, axis=0), c.reshape(-1))
        slow_components = slow_rates(models, x).reshape(-1, 9, 2)
        raw_slow = slow_components.sum(-1)
        reference_components = slow_components[:, 0]
        admissible = np.all(slow_components <= reference_components[:, None, :] + 1e-12, axis=-1)
        slow_upper = policy.upper(np.repeat(subset, 9, axis=0), c.reshape(-1), original=True).reshape(-1, 9)
        admissible &= slow_upper <= np.maximum(2., slow_upper[:, :1]) + 1e-12
        c = np.where(admissible, c, reference_action[:, None])
        slow = np.where(admissible, raw_slow, raw_slow[:, :1])
        low = c[np.arange(len(c)), (slow + projected_config['lambda_low'] * (c - c[:, :1])).argmin(-1)]
        high = c[np.arange(len(c)), (slow + projected_config['lambda_high'] * (c - c[:, :1])).argmin(-1)]
        weight = projected_config['low_policy_weight']
        expected = low if projected_config['lambda_low'] == projected_config['lambda_high'] else weight * low + (1 - weight) * high
        signal = (slow[:, 0] - slow.min(-1) > 1e-12) & (not projected_config['reference_only'])
        expected = np.where(signal, expected, reference_action)
        np.testing.assert_array_equal(policy.act_det(subset), expected)
        np.testing.assert_array_equal(restored.act_det(subset), expected)
        assert np.all(slow_rates(models, public_features(subset, expected)) <= reference_components + 1e-12)
        result = {'passed': True, 'policy_seed': seed, 'controller_config': projected_config,
            'controller_config_sha256': sha(folder / 'controller_config.json'),
            'calibration': calibration, 'unconstrained_fit_mean_action': before_mean,
            'global_fit_mean_action_error': global_error, 'eligible_fit_rows': active_count,
            'eligible_held_rows': eligible_held, 'legacy_changes_without_signal': legacy_without_signal,
            'held_selected_predicted_risk_worse_rows': worse_risk,
            'held_mean_predicted_risk_change': risk_delta_sum / len(obs_held),
            'held_changed_action_rows': changed, 'held_rows': len(obs_held),
            'held_matched_e63_mean_action': float(np.mean(held_original)),
            'held_candidate_mean_action': float(np.mean(held_new)),
            'independent_sklearn_subset_rows': len(subset), 'independent_actions_exact': True,
            'e73_changed_rows_reconstructed': True, 'componentwise_guard_all_held_rows_passed': True,
            'largest_held_component_increase': largest_component_increase.tolist(), 'reference_fallback_all_ineligible_held_rows_exact': True, 'strict_config_reload_exact': True,
            'base_checkpoint_hashes': hashes, 'rate_model_sha256': fit['model_sha256'],
            'capture_provenance': provenance, 'test_evaluated': False, 'new_environment_queries': 0,
            'extra_environment_training_steps': 0, 'base_training_steps': 401280,
            'source_sha256': pinned,
            'forecast_summary_sha256': sha(forecast_path / 'summary.json'),
            'forecast_audit_sha256': sha(forecast_path / 'audit.json'),
            'upper_guard_all_held_rows_sklearn_passed': True,
            'disabled_shield_e74_all_held_actions_exact': True,
            'scope': 'Mean calibration on fitting observations only; held componentwise predicted guard, not counterfactual quality.'}
        (folder / 'fit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        records.append(result)
        print('ANCHORED_PROJECTION_FIT_PASS', seed, projected_config['lambda_low'], projected_config['low_policy_weight'], flush=True)
    summary = {'forecast_summary_sha256': sha(forecast_path / 'summary.json'), 'forecast_audit_sha256': sha(forecast_path / 'audit.json'), 'passed': True, 'policy_seeds': list(range(5)), 'records': records,
        'data_sha256': sha(data), 'baseline_certificate_sha256': sha(args.certificate),
        'test_evaluated': False, 'new_environment_queries': 0, 'source_sha256': pinned}
    (args.out / 'preflight.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('ANCHORED_PROJECTION_PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
