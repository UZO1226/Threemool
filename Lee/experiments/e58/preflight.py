"""Synthetic and calibration-only zero-extension checks before candidate fit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from upper_tail import ROOT, UpperDynamics, interpolate, pit, q, LEVELS

sys.path.insert(0, str(ROOT / 'Lee/experiments/e51'))
from calibrate_blend import prepare, ALPHAS, FrozenDynamics


def synthetic():
    rng = np.random.default_rng(5801)
    central = np.sort(rng.uniform(0, 5, (5, 256, 5)), axis=0)
    values = np.concatenate([central, central[-1:]], axis=0)
    u = rng.random((256, 5))
    for i, endpoint in enumerate([0, .05, .25, .5, .75, .95, .97, .99, 1]):
        u[i] = endpoint
    np.testing.assert_array_equal(interpolate(values, u), q.interpolate(central, u))
    target = rng.uniform(-1, 6, (256, 5))
    target[:5] = central[:, :5].diagonal(axis1=0, axis2=1).T
    np.testing.assert_array_equal(pit(values, target, np.random.default_rng(5802)),
                                  q.pit(central, target, np.random.default_rng(5802)))
    values[-1, :, 2:] += rng.uniform(0, 2, (256, 3))
    expected = np.array([[np.interp(u[i, j], LEVELS, values[:, i, j])
                          for j in range(5)] for i in range(256)])
    np.testing.assert_allclose(interpolate(values, u), expected, rtol=1e-14, atol=1e-14)
    # Upper endpoint atoms include the clipped probability mass above .99.
    top = pit(values, values[-1], np.random.default_rng(5803))
    assert np.all(top[:, 2:] >= .99) and np.all(top <= 1)
    np.testing.assert_array_equal(q.LEVELS, [.05, .25, .5, .75, .95])
    return {'zero_inverse_bitwise': True, 'zero_cdf_bitwise_with_atoms': True,
            'extended_inverse_numpy_reference': True, 'upper_atom_interval': True,
            'historical_levels_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'parent-freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    checks = synthetic()
    parent = FrozenDynamics(args.parent_freeze)
    raw = pd.read_pickle(args.data)
    reference = json.loads(args.reference_summary.read_text())
    banks, old_checks = prepare(parent, raw, reference)
    checks['original_e51_checks'] = old_checks
    dyn = UpperDynamics(parent.model)
    days = reference['calibration_days']
    x, delta = q.build_arrays(raw['data'], days)
    members = np.repeat(np.arange(5), len(x))
    current = np.tile(x[:, :5], (5, 1))
    values = dyn.quantiles(np.tile(x, (5, 1)), members)
    target = np.tile(x[:, :5] + delta, (5, 1))
    for alpha in ALPHAS:
        blended = np.clip(alpha * current[None] + (1 - alpha) * values, 0, dyn.hi)
        bank = pit(blended, target, np.random.default_rng(3801)).reshape(5, len(x), 5)
        bank[..., 1] = .5
        np.testing.assert_array_equal(bank, banks[alpha])
    checks['all_four_zero_banks_bitwise_e51'] = True
    # Complete days, calibration only, original member/noise keys and alpha.
    subset = days[:4]
    batch = q.arrays(raw['data'], subset)
    noise = q.paths(banks[.25], subset, 95, draws=2)
    expected = q.rollout(parent.model, batch, noise, .25, draws=2)
    with patch.object(q, 'interpolate', interpolate):
        actual = q.rollout(dyn, batch, noise, .25, draws=2)
    np.testing.assert_array_equal(actual, expected)
    checks.update({'complete_95_step_zero_rollout_bitwise': True,
                   'calibration_rollout_cases': 40, 'weight_fitting': False,
                   'validation_evaluated': False, 'test_evaluated': False,
                   'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                   'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob('*.py')}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(checks, indent=2, allow_nan=False))
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
