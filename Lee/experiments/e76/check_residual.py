"""Known-answer negative-change, physical-unit and unchanged gate controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from residual_forecast import ROOT, ConstantRate, FastRate, mean_prediction, compiled_mean, margin, upper_admissible, sha
from policy_io import verify_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    previous = json.loads((ROOT / 'Lee/experiments/e75/forecast_check.json').read_text())
    assert previous['passed']
    verify_sources(previous['source_sha256'], ROOT)
    x = np.zeros((3, 12))
    x[:, 2:5] = [1., .8, .2]
    negative = ConstantRate(-.5)
    np.testing.assert_array_equal(FastRate([negative]).raw(x)[:, 0], [-.5] * 3)
    np.testing.assert_array_equal(mean_prediction(negative, x), [.5] * 3)
    np.testing.assert_array_equal(compiled_mean(negative, x), [.5] * 3)
    np.testing.assert_array_equal(compiled_mean(ConstantRate(2.5), x), [3.5] * 3)
    np.testing.assert_array_equal(compiled_mean(ConstantRate(-2.), x), [0.] * 3)
    np.testing.assert_array_equal(compiled_mean(ConstantRate(0), x), x[:, 2:5].max(-1))
    assert margin(np.arange(20) / 10, np.zeros(20)) == 1.9
    np.testing.assert_array_equal(upper_admissible([1.9, 2.1, 2.3, 2.5], [1.6, 1.6, 2.4, 2.4]),
                                 [True, False, True, False])
    paths = [Path(__file__), Path(__file__).with_name('residual_forecast.py'),
        *[ROOT / name for name in previous['source_sha256']]]
    record = {'passed': True, 'negative_changes_preserved': True, 'physical_values_above_one_preserved': True,
        'physical_floor_after_addition': True, 'zero_change_exact_persistence': True,
        'unchanged_margin_and_max_threshold_rule': True, 'test_evaluated': False, 'new_environment_queries': 0,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False))
    print('RESIDUAL_DO_CONTROLS_PASSED', flush=True)


if __name__ == '__main__':
    main()
