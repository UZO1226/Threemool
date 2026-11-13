"""Bound observed-state event calibration attainable inside fixed quantile support."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import FrozenDynamics
from quantile_dynamics import build_arrays


def bound(probability_minimum, probability_maximum, target):
    assert np.all(probability_minimum <= probability_maximum)
    closest = np.clip(target.astype(float), probability_minimum, probability_maximum)
    return (closest - target) ** 2


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    np.testing.assert_allclose(bound(np.array([0, .6, 0]), np.array([.4, 1, .4]),
                                    np.array([True, False, False])), [.36, .36, 0])
    dyn = FrozenDynamics(args.freeze)
    raw = pd.read_pickle(args.data)
    reference = json.loads(args.reference_summary.read_text())
    days = reference['calibration_days']
    assert len(days) == 30 and set(days).isdisjoint(raw['days']['val'] + raw['days']['test'])
    x, delta = build_arrays(raw['data'], days)
    target = x[:, :5] + delta
    values = dyn.model.quantiles(np.tile(x, (5, 1)), np.repeat(np.arange(5), len(x)))
    values = values.reshape(5, 5, len(x), 5)
    result = {
        'test_evaluated': False, 'validation_evaluated': False, 'model_fitted': False,
        'scope': 'Observed-state one-step inner calibration only, not a free-running impossibility bound.',
        'uncertainty_unit': '30 calibration dates, not five independent policy seeds or pooled member rows.',
        'probability_bound': 'Uniform-member lower and upper event probabilities assuming arbitrary within-support noise per row.',
        'brier_bound': 'Mean squared distance of binary target from attainable probability interval. Optimistic; fixed pooled PIT can be more restricted.',
        'diagnostic_does_not_select_settings': True,
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'dates': days, 'transitions': len(x), 'alphas': {}}
    for alpha in (0, .25, .5, .75):
        lower = np.clip(alpha * x[None, :, :5] + (1 - alpha) * values[0], 0, dyn.hi)
        upper = np.clip(alpha * x[None, :, :5] + (1 - alpha) * values[-1], 0, dyn.hi)
        overshoot = np.maximum(target[None] - upper, 0)
        undershoot = np.maximum(lower - target[None], 0)
        row = {
            'member_row_outside_concentration_support_fraction_by_state': ((overshoot > 0) | (undershoot > 0)).mean((0, 1)).tolist(),
            'mean_upper_tail_gap_by_state': overshoot.mean((0, 1)).tolist(),
            'mean_lower_tail_gap_by_state': undershoot.mean((0, 1)).tolist(),
            'events': {}}
        for label in ('nh4', 'do'):
            if label == 'nh4':
                observed = target[:, 0] > 4
                initial = x[:, 0] > 4
                always, possible = lower[..., 0] > 4, upper[..., 0] > 4
            else:
                observed = target[:, 2:].max(-1) > 2
                initial = x[:, 2:5].max(-1) > 2
                always, possible = lower[..., 2:].max(-1) > 2, upper[..., 2:].max(-1) > 2
            pmin, pmax = always.mean(0), possible.mean(0)
            error_bound = bound(pmin, pmax, observed)
            eligible = ~initial
            row['events'][label] = {
                'actual_violations': int(observed.sum()), 'actual_new_onsets': int((observed & eligible).sum()),
                'positive_with_no_member_support': int((observed & (pmax == 0)).sum()),
                'new_onset_with_no_member_support': int((observed & eligible & (pmax == 0)).sum()),
                'negative_with_all_members_forced_violation': int((~observed & (pmin == 1)).sum()),
                'optimistic_ensemble_brier_lower_bound': float(error_bound.mean()),
                'new_onset_subset_brier_lower_bound': float(error_bound[eligible].mean()),
                'eligible_currently_safe_steps': int(eligible.sum())}
        result['alphas'][str(alpha)] = row
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value['events'] for key, value in result['alphas'].items()}), flush=True)


if __name__ == '__main__':
    main()
