"""Calibration-only support diagnostic after the fixed e58 outcome, no tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_tail import UpperDynamics, q


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary = json.loads((args.runs / 'summary.json').read_text())
    assert summary['data_sha256'] == hashlib.sha256(args.data.read_bytes()).hexdigest()
    for name, expected in summary['model_hashes'].items():
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == expected
    raw = pd.read_pickle(args.data)
    days = summary['calibration_days']
    assert len(days) == 30
    assert set(days).isdisjoint(summary['fit_days'] + raw['days']['val'] + raw['days']['test'])
    x, delta = q.build_arrays(raw['data'], days)
    target = x[:, :5] + delta
    alpha = float(summary['selected'].split('_')[2][1:])
    original = q.QuantileDynamics(args.runs)
    dyn = UpperDynamics(original, q.GBMDynamics(args.runs / 'upper_do_0.99.pkl'))
    members = np.repeat(np.arange(5), len(x))
    tiled = np.tile(x, (5, 1))
    central = original.quantiles(tiled, members)
    extended = dyn.quantiles(tiled, members)
    np.testing.assert_array_equal(central, extended[:5])
    result = {'scope': 'Observed-state one-step attainable support on calibration only; not free-running event prediction or policy safety.',
              'selected_alpha_fixed': alpha, 'model_refit': False, 'selection_changed': False,
              'validation_evaluated': False, 'test_evaluated': False,
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(), 'models': {}}
    actual = target[:, 2:].max(1) > 2
    onset = actual & (x[:, 2:5].max(1) <= 2)
    for name, values in [('original_backbone', central), ('conditional_upper', extended)]:
        top = np.clip(alpha * tiled[:, :5] + (1 - alpha) * values[-1], 0, dyn.hi)
        attainable = (top[:, 2:].max(1) > 2).reshape(5, len(x))
        none = ~attainable.any(0)
        result['models'][name] = {
            'actual_do_events': int(actual.sum()), 'actual_new_do_events': int(onset.sum()),
            'events_outside_every_member_upper_support': int((actual & none).sum()),
            'new_events_outside_every_member_upper_support': int((onset & none).sum()),
            'non_events_with_some_member_upper_support': int((~actual & ~none).sum()),
            'per_member_attainable_event_rows': attainable.sum(1).tolist()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
