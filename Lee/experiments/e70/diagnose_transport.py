"""Replay unchanged policies and decompose native air into two exact terms."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e69'))
from budget_policy import load_policy, TrustPolicy
from rate_io import load_base, load_rates
from quantile_env import FrozenDynamics, evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    results = args.workspace.resolve() / 'results'
    study = results / 'e69_study'
    report = json.loads((study / 'comparison.json').read_text())
    assert not report['meets_prespecified_simulation_criterion']
    assert report['policy_seeds'] == list(range(5)) and not report['test_evaluated']
    status = json.loads((study / 'study_status.json').read_text())
    assert status['phase'] == 'complete' and not status['running'] and not status['pending']
    manifest = json.loads((study / 'study_manifest.json').read_text())
    for path, expected in manifest['source_sha256'].items():
        assert sha(Path(path)) == expected
    source = sha(Path(__file__))
    data = args.workspace.resolve() / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    dynamics = FrozenDynamics(freeze)
    args.out.mkdir(parents=True)
    records = []
    factor = 9000 * .25 * 95 / 2
    for seed in range(5):
        assert sha(Path(__file__)) == source
        folder = study / f'comparisons/seed_{seed}'
        pair = json.loads((folder / 'summary.json').read_text())
        audit = json.loads((folder / 'action_audit.json').read_text())
        assert audit['passed'] and audit['comparison_summary_sha256'] == sha(folder / 'summary.json')
        assert pair['candidate_method'] == 'e69_training_air_budget_h4'
        base, config, _, hashes = load_base(results / f'e55_study/policies/new_ppolag_fcsrl/seed_{seed}',
            freeze, results / f'e42_baselines/new_ppolag_fcsrl/seed_{seed}')
        models, fit = load_rates(results / f'e56_study/models/h4/seed_{seed}', config, 4, False)
        assert hashes == pair['base_checkpoint_hashes'] and fit['model_sha256'] == pair['correction_model_sha256']
        assert sha(folder / 'controller_config.json') == pair['controller_config_sha256']
        policies = [TrustPolicy(base, models, 0), load_policy(base, models, folder / 'controller_config.json')]
        occupancies = []
        for control in (0, 1):
            counts = np.zeros(4, dtype=np.int64)
            sums = np.zeros((4, 2))
            def act(obs):
                preserved = obs.copy()
                actions = np.column_stack([policy.act_det(obs) for policy in policies])
                np.testing.assert_array_equal(obs, preserved)
                assert np.isfinite(actions).all() and np.all(np.abs(actions) <= 1)
                regime = 2 * (obs[:, 0] > 4).astype(int) + (obs[:, 2:5].max(-1) > 2).astype(int)
                for group in range(4):
                    selected = regime == group
                    counts[group] += int(selected.sum())
                    sums[group] += actions[selected].sum(0)
                return actions[:, control]
            summary, rows = evaluate(data, dynamics, act_batch=act)
            saved_path = results / f'e63_study/comparisons/seed_{seed}/candidate_native.json' if control == 0 else folder / 'candidate_native.json'
            assert json.loads(saved_path.read_text()) == {'summary': summary, 'rows': rows}
            assert counts.sum() == 152000
            means = sums.sum(0) / counts.sum()
            np.testing.assert_allclose(factor * (means[control] + 1), summary['air_m3'], rtol=1e-12, atol=1e-8)
            occupancies.append({'control': 'e63' if control == 0 else 'e69', 'action_means': means.tolist(),
                'regime_counts': counts.tolist(),
                'conditional_action_means': [(sums[group] / counts[group]).tolist() if counts[group] else None for group in range(4)],
                'native_summary': summary, 'saved_native_sha256': sha(saved_path), 'native_replay_exact': True})
            print('AIR_TRANSPORT_REPLAY', seed, control, flush=True)
        old, new = [np.array(item['action_means']) for item in occupancies]
        actual = factor * (new[1] - old[0])
        mapping_old, mapping_new = factor * (old[1] - old[0]), factor * (new[1] - new[0])
        occupancy_new, occupancy_old = factor * (new[1] - old[1]), factor * (new[0] - old[0])
        np.testing.assert_allclose([mapping_old + occupancy_new, mapping_new + occupancy_old], actual, atol=1e-8, rtol=0)
        mapping, occupancy = (mapping_old + mapping_new) / 2, (occupancy_old + occupancy_new) / 2
        np.testing.assert_allclose(mapping + occupancy, actual, atol=1e-8, rtol=0)
        record = {'policy_seed': seed, 'occupancies': occupancies,
            'actual_native_air_difference_m3': float(actual), 'symmetric_mapping_component_m3': float(mapping),
            'symmetric_occupancy_component_m3': float(occupancy),
            'mapping_on_e63_occupancy_m3': float(mapping_old), 'mapping_on_e69_occupancy_m3': float(mapping_new),
            'occupancy_under_e63_action_m3': float(occupancy_old), 'occupancy_under_e69_action_m3': float(occupancy_new),
            'base_checkpoint_hashes': hashes, 'rate_model_sha256': fit['model_sha256'],
            'controller_config_sha256': pair['controller_config_sha256'], 'test_evaluated': False}
        records.append(record)
        (args.out / f'seed_{seed}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    indices = np.random.default_rng(2026).integers(0, 5, size=(10000, 5))
    components = {}
    for field in ('actual_native_air_difference_m3', 'symmetric_mapping_component_m3', 'symmetric_occupancy_component_m3'):
        values = np.array([record[field] for record in records])
        components[field] = {'mean': float(values.mean()), 'policy_seed_values': values.tolist(),
            'paired_seed_bootstrap95_ci': np.quantile(values[indices].mean(1), [.025, .975]).tolist()}
    result = {'passed': True, 'policy_seeds': list(range(5)), 'records': records, 'air_components': components,
        'environment_evaluation_transitions': 5 * 2 * 152000, 'environment_training_steps': 0,
        'test_evaluated': False, 'candidate_selection': False, 'source_sha256': source,
        'initial_report_sha256': sha(study / 'comparison.json'),
        'scope': 'Exact native model decomposition, not a new controller, adoption or physical causal effect.'}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('AIR_TRANSPORT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
