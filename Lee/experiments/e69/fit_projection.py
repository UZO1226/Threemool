"""Calibrate only fitting-state mean action, then diagnose held choices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from budget_policy import ROOT, BudgetPolicy, TrustPolicy, calibrate, configuration, load_policy, project, public_features
from rate_io import load_base, load_rates
from rate_model import slow_rates
from streams import read_stream
from policy_io import verify_sources

sys.path.append(str(ROOT / 'Lee/experiments/e67'))
from radius_policy import RadiusPolicy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'certificate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    certificate = json.loads(args.certificate.read_text())
    assert certificate['passed'] and certificate['policy_seeds'] == list(range(5))
    assert sha(data) == certificate['data_sha256']
    prior = json.loads((ROOT / 'Lee/experiments/e67/preflight.json').read_text())
    verify_sources({'radius_policy.py': prior['source_sha256']['radius_policy.py']}, ROOT / 'Lee/experiments/e67')
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
        zero = BudgetPolicy(base, models, configuration(0, 0, 1))
        candidates, risks, targets = [], [], []
        for start in range(0, len(obs_fit), 2048):
            c, r = zero.geometry(obs_fit[start:start + 2048])
            old_action = c[np.arange(len(c)), r[:, :5].argmin(-1)]
            candidates.append(c)
            risks.append(r)
            targets.append(old_action)
        candidates, risks = np.concatenate(candidates), np.concatenate(risks)
        target = float(np.concatenate(targets).mean())
        projected_config, calibration = calibrate(candidates, risks, target)
        before_mean = float(project(candidates, risks, zero.config).mean())
        del candidates, risks, targets
        folder = args.out / f'seed_{seed}'
        folder.mkdir()
        policy = BudgetPolicy(base, models, projected_config)
        policy.save_config(folder / 'controller_config.json')
        restored = load_policy(base, models, folder / 'controller_config.json')
        old, radius = TrustPolicy(base, models, 0), RadiusPolicy(base, models)
        changed = 0
        held_original, held_new = [], []
        for start in range(0, len(obs_held), 2048):
            obs = obs_held[start:start + 2048]
            unchanged = obs.copy()
            c, r = policy.geometry(obs)
            action = project(c, r, projected_config)
            np.testing.assert_array_equal(project(c, r, zero.config), radius.act_det(obs))
            previous = old.act_det(obs)
            np.testing.assert_array_equal(previous, c[np.arange(len(c)), r[:, :5].argmin(-1)])
            assert np.all(np.abs(action - c[:, 0]) <= .5 + 1e-12)
            assert np.isfinite(action).all() and np.all(np.abs(action) <= 1)
            np.testing.assert_array_equal(obs, unchanged)
            changed += int(np.count_nonzero(action != previous))
            held_original.extend(previous.tolist())
            held_new.extend(action.tolist())
        subset = obs_held[np.linspace(0, len(obs_held) - 1, 1024, dtype=int)]
        original = np.asarray(base.act_det(subset)).reshape(-1)
        c = np.clip(original[:, None] + [0, -.125, .125, -.25, .25, -.375, .375, -.5, .5], -1, 1)
        x = public_features(np.repeat(subset, 9, axis=0), c.reshape(-1))
        slow = slow_rates(models, x).sum(-1).reshape(-1, 9)
        low = c[np.arange(len(c)), (slow + projected_config['lambda_low'] * (c - c[:, :1])).argmin(-1)]
        high = c[np.arange(len(c)), (slow + projected_config['lambda_high'] * (c - c[:, :1])).argmin(-1)]
        weight = projected_config['low_policy_weight']
        expected = low if projected_config['lambda_low'] == projected_config['lambda_high'] else weight * low + (1 - weight) * high
        np.testing.assert_array_equal(policy.act_det(subset), expected)
        np.testing.assert_array_equal(restored.act_det(subset), expected)
        result = {'passed': True, 'policy_seed': seed, 'controller_config': projected_config,
            'controller_config_sha256': sha(folder / 'controller_config.json'),
            'calibration': calibration, 'unconstrained_fit_mean_action': before_mean,
            'held_changed_action_rows': changed, 'held_rows': len(obs_held),
            'held_e63_native_mean_action': float(np.mean(held_original)),
            'held_candidate_mean_action': float(np.mean(held_new)),
            'independent_sklearn_subset_rows': len(subset), 'independent_actions_exact': True,
            'zero_multiplier_e67_identity_all_held_rows': True, 'strict_config_reload_exact': True,
            'base_checkpoint_hashes': hashes, 'rate_model_sha256': fit['model_sha256'],
            'capture_provenance': provenance, 'test_evaluated': False, 'new_environment_queries': 0,
            'extra_environment_training_steps': 0, 'base_training_steps': 401280,
            'source_sha256': pinned,
            'scope': 'Mean calibration on fitting observations only; held geometry, not counterfactual quality.'}
        (folder / 'fit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        records.append(result)
        print('PROJECTION_FIT_PASS', seed, projected_config['lambda_low'], projected_config['low_policy_weight'], flush=True)
    summary = {'passed': True, 'policy_seeds': list(range(5)), 'records': records,
        'data_sha256': sha(data), 'baseline_certificate_sha256': sha(args.certificate),
        'test_evaluated': False, 'new_environment_queries': 0, 'source_sha256': pinned}
    (args.out / 'preflight.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('PROJECTION_PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
