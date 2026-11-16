"""Secondary same-base/rate-model contrast to completed original h4 correction."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'previous', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pairs = []
    for seed in range(5):
        current = json.loads((args.study / f'comparisons/seed_{seed}/summary.json').read_text())
        previous = json.loads((args.previous / f'comparisons/h4/seed_{seed}/summary.json').read_text())
        assert current['policy_seed'] == previous['policy_seed'] == seed
        assert current['candidate_method'] == 'ppolag_fcsrl_risk_only_h4'
        assert previous['candidate_method'] == 'ppolag_fcsrl_rate_h4'
        for key in ('freeze_canonical_sha256', 'baseline_checkpoint_hashes', 'candidate_checkpoint_hashes',
                    'correction_model_sha256', 'baseline_native', 'training_steps'):
            assert current[key] == previous[key]
        for pair in (current, previous):
            assert pair['training_steps'] == 401280 and not pair['test_evaluated'] and not pair['smoke_only']
            assert pair['checkpoint_replay_verified'] and pair['matching']['matched_within_one_percent']
            assert abs(pair['candidate_matched']['air_m3'] - pair['baseline_native']['air_m3']) <= .01 * pair['baseline_native']['air_m3']
        assert not current['pipeline_smoke_only']
        a, b = current['candidate_matched']['air_m3'], previous['candidate_matched']['air_m3']
        assert abs(a - b) <= .01 * b
        pairs.append((current, previous))
    indices = np.random.default_rng(2026).integers(0, 5, size=(10000, 5))
    metrics = {}
    for mode in ('candidate_native', 'candidate_matched'):
        metrics[mode] = {}
        for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct'):
            current = np.array([a[mode][key] for a, _ in pairs])
            previous = np.array([b[mode][key] for _, b in pairs])
            difference = current - previous
            metrics[mode][key] = {'risk_only_mean': float(current.mean()), 'original_penalty_mean': float(previous.mean()),
                'risk_only_sd': float(current.std(ddof=1)), 'original_penalty_sd': float(previous.std(ddof=1)),
                'paired_difference': difference.tolist(), 'paired_mean_difference': float(difference.mean()),
                'paired_seed_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    result = {'policy_seeds': list(range(5)), 'same_base_rate_models_and_reference_air_verified': True,
              'metrics': metrics, 'test_evaluated': False, 'new_training': False,
              'scope': 'Secondary reused-five-seed contrast, cannot rescue primary failure versus original FCSRL.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('PENALTY_CONTRAST_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
