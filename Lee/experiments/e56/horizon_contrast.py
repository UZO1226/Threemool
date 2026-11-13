"""Secondary horizon4 minus loss-matched horizon1 at common FCSRL air."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    indexed = {}
    for horizon in (1, 4):
        pairs = []
        for seed in range(5):
            row = json.loads((args.study / f'comparisons/h{horizon}/seed_{seed}/summary.json').read_text())
            assert row['policy_seed'] == seed and row['correction_horizon'] == horizon
            assert row['candidate_method'] == f'ppolag_fcsrl_rate_h{horizon}'
            assert row['baseline_method'] == 'ppolag_fcsrl' and row['kind'] == 'agent'
            assert row['training_steps'] == 401280 and not row['smoke_only'] and not row['test_evaluated']
            assert row['checkpoint_replay_verified'] and row['candidate_matched'] is not None
            assert row['evaluation_environment'] == row['baseline_training_environment'] == row['candidate_training_environment'] == 'new'
            pairs.append(row)
        indexed[horizon] = pairs
    assert len({row['freeze_canonical_sha256'] for rows in indexed.values() for row in rows}) == 1
    matches, mutual, references = [], [], []
    for seed in range(5):
        one, four = indexed[1][seed], indexed[4][seed]
        assert one['baseline_checkpoint_hashes'] == four['baseline_checkpoint_hashes']
        assert one['candidate_checkpoint_hashes'] == four['candidate_checkpoint_hashes'] == one['baseline_checkpoint_hashes']
        assert one['baseline_native'] == four['baseline_native']
        models = [json.loads((args.study / f'models/h{h}/seed_{seed}/fit_summary.json').read_text()) for h in (1, 4)]
        for key in ('capture_file_sha256', 'fit_updates', 'held_updates', 'base_checkpoint_hashes', 'policy_seed', 'data_sha256', 'loss', 'no3_retained'):
            assert models[0][key] == models[1][key]
        assert all(not model['smoke_only'] and model['checks']['passed'] for model in models)
        target = one['baseline_native']['air_m3']
        references.append(target)
        matches.append(all(row['matching']['matched_within_one_percent'] and
            abs(row['candidate_matched']['air_m3'] - target) <= max(.01 * target, 1e-9) for row in (one, four)))
        mutual.append(abs(one['candidate_matched']['air_m3'] - four['candidate_matched']['air_m3']) <= max(.01 * target, 1e-9))
    indices = np.random.default_rng(2026).integers(0, 5, size=(10000, 5))
    metrics = {}
    for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'reward', 'days_within_limit_pct'):
        one = np.array([row['candidate_matched'][key] for row in indexed[1]])
        four = np.array([row['candidate_matched'][key] for row in indexed[4]])
        assert np.isfinite(one).all() and np.isfinite(four).all()
        difference = four - one
        metrics[key] = {'horizon1_values': one.tolist(), 'horizon4_values': four.tolist(),
                        'horizon1_mean': float(one.mean()), 'horizon4_mean': float(four.mean()),
                        'horizon1_sd': float(one.std(ddof=1)), 'horizon4_sd': float(four.std(ddof=1)),
                        'paired_difference': difference.tolist(), 'paired_mean_difference': float(difference.mean()),
                        'paired_seed_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    checks = {'all_variants_match_reference_air': all(matches), 'all_variants_match_each_other_air': all(mutual),
              'cost_interval_below_zero': metrics['cost']['paired_seed_bootstrap95_ci'][1] < 0,
              'nh4_interval_below_zero': metrics['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1] < 0,
              'do_interval_not_above_zero': metrics['do_viol_pct']['paired_seed_bootstrap95_ci'][1] <= 0}
    result = {'policy_seeds': list(range(5)), 'reference_air_m3': references, 'metrics': metrics,
              'checks': checks, 'longer_horizon_specific_benefit_supported': all(checks.values()),
              'same_base_checkpoints_and_training_captures_verified': True, 'test_evaluated': False,
              'scope': 'Secondary horizon4 minus horizon1 regression control, common FCSRL air, five original policy seeds. Reused validation, no plant claim. This control cannot replace a failed primary.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('HORIZON_CONTRAST_COMPLETE', json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
