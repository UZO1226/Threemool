"""Compare the calibrated controller with its completed radius0.5 control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'previous', 'smoke', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for folder in (args.study, args.previous):
        status = json.loads((folder / 'study_status.json').read_text())
        assert status['phase'] == 'complete' and not status['running'] and not status['pending']
    smoke = json.loads((args.smoke / 'action_audit.json').read_text())
    assert smoke['passed'] and smoke['zero_multiplier_e67_full_native_exact']
    pairs, sources = [], {}
    for seed in range(5):
        current_path = args.study / f'comparisons/seed_{seed}/summary.json'
        old_path = args.previous / f'comparisons/seed_{seed}/summary.json'
        current, old = [json.loads(path.read_text()) for path in (current_path, old_path)]
        assert current['policy_seed'] == old['policy_seed'] == seed
        assert current['candidate_method'] == 'e69_training_air_budget_h4'
        assert old['candidate_method'] == 'e67_larger_radius_h4'
        for key in ('freeze_canonical_sha256', 'base_checkpoint_hashes', 'correction_model_sha256',
                    'baseline_native', 'baseline_fixed_offset', 'training_steps'):
            assert current[key] == old[key]
        for path, pair in ((current_path, current), (old_path, old)):
            assert pair['training_steps'] == 401280 and not pair['test_evaluated'] and not pair['smoke_only']
            assert not pair['pipeline_smoke_only'] and pair['checkpoint_replay_verified']
            assert pair['matching']['matched_within_one_percent']
            target, actual = pair['baseline_native']['air_m3'], pair['candidate_matched']['air_m3']
            assert abs(actual - target) <= .01 * target
            audit_path = path.parent / 'action_audit.json'
            audit = json.loads(audit_path.read_text())
            assert audit['passed'] and audit['comparison_summary_sha256'] == sha(path)
            assert set(audit['replays']) == {'native', 'matched'}
            assert all(item['action_rows'] == 152000 and item['trajectory_replay_exact'] and
                       item['independent_sklearn_actions_exact'] for item in audit['replays'].values())
            sources[str(path)] = sha(path)
            sources[str(audit_path)] = sha(audit_path)
        a, b = current['candidate_matched']['air_m3'], old['candidate_matched']['air_m3']
        assert abs(a - b) <= .01 * b
        configs = [json.loads((path.parent / 'controller_config.json').read_text()) for path in (current_path, old_path)]
        for key in ('candidate_offsets', 'horizon', 'hard_action_radius', 'observation_dimension',
                    'rate_output_clip', 'action_clip', 'tie_rule'):
            assert configs[0][key] == configs[1][key]
        pairs.append((current, old))
    indices = np.random.default_rng(2026).integers(0, 5, size=(10000, 5))
    metrics = {}
    for mode in ('candidate_native', 'candidate_matched'):
        metrics[mode] = {}
        for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct'):
            new = np.array([pair[0][mode][key] for pair in pairs])
            old = np.array([pair[1][mode][key] for pair in pairs])
            delta = new - old
            metrics[mode][key] = {'projection_mean': float(new.mean()), 'unconstrained_control_mean': float(old.mean()),
                'projection_sd': float(new.std(ddof=1)), 'control_sd': float(old.std(ddof=1)),
                'paired_difference': delta.tolist(), 'paired_mean_difference': float(delta.mean()),
                'paired_seed_bootstrap95_ci': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    result = {'policy_seeds': list(range(5)), 'same_base_models_grid_and_primary_reference_verified': True,
        'metrics': metrics, 'test_evaluated': False, 'new_training': False, 'source_sha256': sources,
        'generator_sha256': sha(Path(__file__)),
        'scope': 'Secondary exploratory mechanism contrast only. Both policies matched air to the same e63 references, not directly to each other. Cannot rescue failed primary e63 criterion.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('PROJECTION_CONTROL_CONTRAST_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
