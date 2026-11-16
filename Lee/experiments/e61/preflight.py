"""Calibration-only identity and causal-input preflight for context PIT paths."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from context_paths import initial_context, fit_scale, donor_pools, paths

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e38'))
import quantile_dynamics as q
sys.path.insert(0, str(ROOT / 'Lee/experiments/e58'))
from upper_tail import UpperDynamics, pit as upper_pit


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e58', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    reference = json.loads((args.e51 / 'summary.json').read_text())
    raw = pd.read_pickle(args.data)
    fit, cal = reference['fit_days'], reference['calibration_days']
    assert len(fit) == 158 and len(cal) == 30
    assert set(fit).isdisjoint(cal)
    assert set(fit + cal).isdisjoint(raw['days']['val'] + raw['days']['test'])
    fitting = initial_context(q.arrays(raw['data'], fit))
    batch = q.arrays(raw['data'], cal)
    donors = initial_context(batch)
    mean, sd, floored = fit_scale(fitting)
    changed = tuple(array.copy() for array in batch)
    changed[0][:, 1:] += 1000
    changed[1][:, 1:] -= 1000
    changed[2][:] = 0
    np.testing.assert_array_equal(initial_context(changed), donors)
    pools = donor_pools(donors, donors, cal, cal, mean, sd)
    changed_pools = donor_pools(initial_context(changed), donors, cal, cal, mean, sd)
    for index, (pool, other) in enumerate(zip(pools, changed_pools)):
        assert len(pool) == 5 and index not in pool and np.all(np.diff(pool) > 0)
        np.testing.assert_array_equal(pool, other)
    global_loo = donor_pools(donors, donors, cal, cal, mean, sd, k=None)
    assert all(len(pool) == 29 and i not in pool for i, pool in enumerate(global_loo))
    tied = donor_pools(np.zeros((1, 6)), np.zeros((30, 6)), ['query'], cal,
                       np.zeros(6), np.ones(6), k=5)
    np.testing.assert_array_equal(tied[0], np.arange(5))
    full = donor_pools(donors[:4], donors, cal[:4], cal, mean, sd, k=None, exclude_self=False)
    x, delta = q.build_arrays(raw['data'], cal)
    members = np.repeat(np.arange(5), len(x))
    current = np.tile(x[:, :5], (5, 1))
    target = np.tile(x[:, :5] + delta, (5, 1))
    bank_checksums = {}
    for directory in (args.e51, args.e58):
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['data_sha256'] == digest and summary['test_evaluated'] is False
        assert summary['fit_days'] == fit and summary['calibration_days'] == cal
        for name, expected in summary['model_hashes'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        bank = np.load(directory / 'pit_bank.npz')['pit']
        backbone = q.QuantileDynamics(directory)
        if directory == args.e58:
            dyn = UpperDynamics(backbone, q.GBMDynamics(directory / 'upper_do_0.99.pkl'))
            cdf = upper_pit
        else:
            dyn, cdf = backbone, q.pit
        values = dyn.quantiles(np.tile(x, (5, 1)), members)
        mixed = np.clip(.25 * current[None] + .75 * values, 0, dyn.hi)
        rebuilt = cdf(mixed, target, np.random.default_rng(3801)).reshape(5, len(x), 5)
        rebuilt[..., 1] = .5
        np.testing.assert_array_equal(rebuilt, bank)
        bank_checksums['e58' if directory == args.e58 else 'e51'] = hashlib.sha256(bank.tobytes()).hexdigest()
        np.testing.assert_array_equal(paths(bank, cal[:4], full, draws=2), q.paths(bank, cal[:4], 95, draws=2))
        selected = paths(bank, cal, pools, draws=2)
        np.testing.assert_array_equal(selected, paths(bank, cal, pools, draws=2))
        # Independent scalar RNG and donor reconstruction, every preflight case.
        for member in range(5):
            for draw in range(2):
                for index, day in enumerate(cal):
                    seed = int.from_bytes(hashlib.sha256(f'e38:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                    donor = pools[index][np.random.default_rng(seed).integers(0, 5)]
                    case = (member * 2 + draw) * len(cal) + index
                    np.testing.assert_array_equal(selected[case], bank[member, donor * 95 + np.arange(95)])
    result = {'data_sha256': digest, 'full_pool_original_paths_bitwise_both_banks': True,
              'future_state_exogenous_air_invariance': True, 'leave_self_out_five_and_global_pools': True,
              'deterministic_tie_and_chronological_order': True, 'independent_donor_and_tuple_copy_exact': True,
              'reproducibility_exact': True, 'fit_context_count': len(fitting),
              'both_selected_banks_reconstructed_bitwise': True,
              'bank_array_sha256': bank_checksums,
              'feature_order': ['nh4', 'do1', 'do2', 'do3', 'temperature', 'flow'],
              'fit_context_mean': mean.tolist(), 'fit_context_sd': sd.tolist(),
              'sd_floor_used': floored.tolist(), 'weight_fitting': False,
              'validation_evaluated': False, 'test_evaluated': False,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
