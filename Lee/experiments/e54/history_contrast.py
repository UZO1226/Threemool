"""Secondary history/repeated comparison at the same original FCSRL reference air."""
import argparse
import json
from pathlib import Path

import numpy as np


def contrast(history_pairs, repeated_pairs):
    def indexed(pairs, method):
        assert len(pairs) == 5 and sorted(row['policy_seed'] for row in pairs) == list(range(5))
        for row in pairs:
            assert row['kind'] == 'agent' and row['baseline_method'] == 'ppolag_fcsrl' and row['candidate_method'] == method
            assert row['evaluation_environment'] == row['baseline_training_environment'] == row['candidate_training_environment'] == 'new'
            assert row['training_steps'] == 401280 and not row['smoke_only'] and not row['test_evaluated']
            assert row['checkpoint_replay_verified'] and row['candidate_matched'] is not None
        return {row['policy_seed']: row for row in pairs}
    history = indexed(history_pairs, 'ppolag_fcsrl_history')
    repeated = indexed(repeated_pairs, 'ppolag_fcsrl_repeated')
    assert len({row['freeze_canonical_sha256'] for row in history_pairs + repeated_pairs}) == 1
    references, matched, mutual = [], [], []
    for seed in range(5):
        h, r = history[seed], repeated[seed]
        assert h['baseline_checkpoint_hashes'] == r['baseline_checkpoint_hashes']
        np.testing.assert_allclose(h['baseline_native']['air_m3'], r['baseline_native']['air_m3'], rtol=1e-12)
        target = h['baseline_native']['air_m3']
        references.append(target)
        matched.append(all(row['matching']['matched_within_one_percent'] and
            abs(row['candidate_matched']['air_m3'] - target) <= max(target * .01, 1e-9) for row in (h, r)))
        mutual.append(abs(h['candidate_matched']['air_m3'] - r['candidate_matched']['air_m3']) <= max(target * .01, 1e-9))
    indices = np.random.default_rng(2026).integers(0, 5, size=(10000, 5))
    metrics = {}
    for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'reward', 'days_within_limit_pct'):
        h = np.array([history[seed]['candidate_matched'][key] for seed in range(5)])
        r = np.array([repeated[seed]['candidate_matched'][key] for seed in range(5)])
        assert np.isfinite(h).all() and np.isfinite(r).all()
        difference = h - r
        metrics[key] = {'history_values': h.tolist(), 'repeated_values': r.tolist(),
            'history_mean': float(h.mean()), 'repeated_mean': float(r.mean()),
            'history_sd': float(h.std(ddof=1)), 'repeated_sd': float(r.std(ddof=1)),
            'paired_difference': difference.tolist(),
            'paired_seed_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    checks = {'all_variants_match_reference_air': all(matched), 'all_variants_match_each_other_air': all(mutual),
        'cost_interval_below_zero': metrics['cost']['paired_seed_bootstrap95_ci'][1] < 0,
        'nh4_interval_below_zero': metrics['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1] < 0,
        'do_interval_not_above_zero': metrics['do_viol_pct']['paired_seed_bootstrap95_ci'][1] <= 0}
    return {'policy_seeds': list(range(5)), 'reference_fcsrl_air_m3': references, 'metrics': metrics,
        'checks': checks, 'history_specific_benefit_supported_in_simulation': all(checks.values()),
        'test_evaluated': False, 'freeze_canonical_sha256': history[0]['freeze_canonical_sha256'],
        'scope': 'Five paired policy seeds at common original-FCSRL air, exploratory reused validation, no real-plant claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('history-pairs', 'repeated-pairs'):
        parser.add_argument('--' + name, type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = contrast([json.loads(path.read_text()) for path in args.history_pairs],
                      [json.loads(path.read_text()) for path in args.repeated_pairs])
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['checks']), flush=True)
