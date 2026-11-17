"""Describe fitted e71 component-risk trades without another environment query."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e71'))
from signal_policy import SignalPolicy, public_features
from rate_io import load_base, load_rates
from rate_model import slow_rates
from streams import read_stream

TOL = 1e-12
OFFSETS = np.array([0, -.125, .125, -.25, .25, -.375, .375, -.5, .5])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    results = workspace / 'results'
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    initial_path, fresh_path = results / 'e71_study/comparison.json', results / 'e71_fresh_study/comparison.json'
    assert read(initial_path)['meets_prespecified_simulation_criterion']
    assert not read(fresh_path)['meets_prespecified_simulation_criterion']
    args.out.mkdir(parents=True)
    records = []
    for seed in range(10):
        fresh = seed >= 5
        run = results / (f'e68_fresh_baselines/captured/new_ppolag_fcsrl/seed_{seed}' if fresh else f'e55_study/policies/new_ppolag_fcsrl/seed_{seed}')
        original = results / (f'e68_fresh_baselines/original/new_ppolag_fcsrl/seed_{seed}' if fresh else f'e42_baselines/new_ppolag_fcsrl/seed_{seed}')
        correction = results / (f'e68_fresh_rates/models/h4/seed_{seed}' if fresh else f'e56_study/models/h4/seed_{seed}')
        calibration_folder = results / ('e71_fresh_fit' if fresh else 'e71_signal_fit') / f'seed_{seed}'
        calibration_path = calibration_folder / 'fit_summary.json'
        calibration = read(calibration_path)
        base, config, _, hashes = load_base(run, freeze, original)
        models, fit = load_rates(correction, config, 4, False)
        assert calibration['passed'] and calibration['policy_seed'] == seed
        assert hashes == calibration['base_checkpoint_hashes'] == fit['base_checkpoint_hashes']
        assert fit['model_sha256'] == calibration['rate_model_sha256']
        policy = SignalPolicy(base, models, calibration['controller_config'])
        fitting, held, provenance = read_stream(run, data, 4, False)
        del fitting
        obs = held['x'][:, :11]
        assert len(obs) == 85120
        counts = {key: 0 for key in ('changed', 'sum_risk_worse', 'nh4_risk_increase', 'do_risk_increase',
            'nh4_increase_do_decrease', 'do_increase_nh4_decrease', 'both_components_nonincreasing',
            'expanded_sum_gain', 'component_guard_gain_available', 'guard_lower_air_available', 'guard_higher_air_available')}
        delta_sum = np.zeros(2)
        for start in range(0, len(obs), 2048):
            batch = obs[start:start + 2048]
            native = np.asarray(base.act_det(batch)).reshape(-1)
            candidates = np.clip(native[:, None] + OFFSETS, -1, 1)
            x = public_features(np.repeat(batch, 9, axis=0), candidates.reshape(-1))
            components = policy.risk.predict(x).reshape(-1, 9, 2)
            total = components.sum(-1)
            indices = total[:, :5].argmin(-1)
            reference_action = candidates[np.arange(len(batch)), indices]
            reference_rate = components[np.arange(len(batch)), indices]
            action = policy.act_det(batch)
            actual_rate = policy.risk.predict(public_features(batch, action))
            delta = actual_rate - reference_rate
            delta_sum += delta.sum(0)
            n_up, d_up = delta[:, 0] > TOL, delta[:, 1] > TOL
            n_down, d_down = delta[:, 0] < -TOL, delta[:, 1] < -TOL
            gain = reference_rate.sum(-1) - total.min(-1) > TOL
            admissible = np.all(components <= reference_rate[:, None, :] + TOL, axis=-1)
            better = reference_rate.sum(-1)[:, None] - total > TOL
            feasible = admissible & better
            masks = {'changed': action != reference_action, 'sum_risk_worse': delta.sum(-1) > TOL,
                'nh4_risk_increase': n_up, 'do_risk_increase': d_up,
                'nh4_increase_do_decrease': n_up & d_down, 'do_increase_nh4_decrease': d_up & n_down,
                'both_components_nonincreasing': ~n_up & ~d_up, 'expanded_sum_gain': gain,
                'component_guard_gain_available': feasible.any(-1),
                'guard_lower_air_available': (feasible & (candidates < reference_action[:, None])).any(-1),
                'guard_higher_air_available': (feasible & (candidates > reference_action[:, None])).any(-1)}
            for key, mask in masks.items():
                counts[key] += int(mask.sum())
        assert counts['changed'] == calibration['held_changed_action_rows']
        assert counts['sum_risk_worse'] == calibration['held_selected_predicted_risk_worse_rows']
        np.testing.assert_allclose(delta_sum.sum() / len(obs), calibration['held_mean_predicted_risk_change'], rtol=0, atol=1e-12)
        subset = obs[np.linspace(0, len(obs) - 1, 1024, dtype=int)]
        action = policy.act_det(subset)
        probe = public_features(subset, action)
        np.testing.assert_allclose(policy.risk.predict(probe), slow_rates(models, probe), atol=1e-12, rtol=1e-12)
        record = {'policy_seed': seed, 'block': 'exposed_fresh' if fresh else 'original',
            'held_rows': len(obs), 'counts': counts, 'mean_predicted_component_change': delta_sum.tolist(),
            'component_change_denominator': len(obs), 'independent_sklearn_probe_rows': len(subset),
            'passed': True, 'calibration_sha256': sha(calibration_path), 'rate_model_sha256': fit['model_sha256'],
            'base_checkpoint_hashes': hashes, 'capture_provenance': provenance,
            'test_evaluated': False, 'new_environment_queries': 0}
        record['mean_predicted_component_change'] = (delta_sum / len(obs)).tolist()
        (args.out / f'seed_{seed}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        records.append(record)
        print('COMPONENT_DIAGNOSTIC_PASS', seed, counts, flush=True)
    summary = {'passed': True, 'policy_seeds': list(range(10)), 'records': records,
        'initial_report_sha256': sha(initial_path), 'fresh_report_sha256': sha(fresh_path),
        'source_sha256': sha(Path(__file__)), 'new_environment_queries': 0,
        'test_evaluated': False, 'new_model_fitting': False,
        'scope': 'Held logged-state predicted components and guard feasibility, not observed candidate outcomes, causality or policy improvement.'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('COMPONENT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
