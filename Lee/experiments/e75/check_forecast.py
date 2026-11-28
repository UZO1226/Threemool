"""Known physical-unit, empirical-margin and threshold-admissibility controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from do_forecast import ROOT, ConstantRate, compiled_mean, margin, upper_admissible, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    x = np.zeros((3, 12))
    np.testing.assert_array_equal(compiled_mean(ConstantRate(2.5), x), [2.5] * 3)
    np.testing.assert_array_equal(compiled_mean(ConstantRate(-.5), x), [0.] * 3)
    assert margin(np.arange(20) / 10, np.zeros(20)) == 1.9
    assert margin(np.zeros(20), np.ones(20)) == 0
    np.testing.assert_array_equal(upper_admissible([1.9, 2.1, 2.3, 2.5], [1.6, 1.6, 2.4, 2.4]),
                                 [True, False, True, False])
    result = diagnostics(np.array([1., 3., 3.]), np.array([1., 2., 3.]), np.array([2., 2., 4.]), np.array([1., 1., 3.]))
    assert result['false_negatives'] == 1 and result['onset_events'] == 1 and result['onset_false_negatives'] == 1
    paths = [Path(__file__), Path(__file__).with_name('do_forecast.py'), Path(__file__).with_name('forecast_data.py'),
             ROOT / 'Lee/experiments/e56/rate_model.py', ROOT / 'Lee/experiments/e56/targets.py', ROOT / 'Lee/experiments/e55/fast_cost.py']
    record = {'passed': True, 'physical_values_above_one_preserved': True, 'nonnegative_mean': True,
        'higher_order_margin_and_nonnegative_floor': True, 'reference_always_feasible_max_rule': True,
        'next_event_and_onset_controls': True, 'test_evaluated': False, 'new_environment_queries': 0,
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False))
    print('DO_FORECAST_CONTROLS_PASSED', flush=True)


if __name__ == '__main__':
    main()
