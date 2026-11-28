"""Calibration-only donor, causal-input and independent copying checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exogenous_blocks import CONTEXT, donor_pools, fit_scale, paths


def digest(path):
    content = path.read_bytes()
    if path.suffix in ('.py', '.json'):
        content = content.replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    workspace = args.workspace.resolve()
    freeze = json.loads(args.freeze.read_text())
    for name, expected in freeze['frozen_files_sha256'].items():
        assert digest(workspace / name) == expected, name
    parent = workspace / freeze['environment_run']
    source = json.loads((parent / 'summary.json').read_text())
    data_path = workspace / 'kaggle_data/darrow_15min.pkl'
    data = pd.read_pickle(data_path)
    fit, cal = source['fit_days'], source['calibration_days']
    assert len(fit) == 158 and len(cal) == 30 and set(fit).isdisjoint(cal)
    assert set(fit + cal) == set(data['days']['train'])
    assert set(fit + cal).isdisjoint(data['days']['val'] + data['days']['test'])
    def context(days):
        return np.stack([data['data'].loc[day][CONTEXT].to_numpy(float)[:95] for day in days])
    fitting, donors = context(fit), context(cal)
    mean, sd, floor = fit_scale(fitting)
    pools = donor_pools(donors, donors, cal, cal, mean, sd)
    for i, day in enumerate(cal):
        eligible = [(j, step) for j, other in enumerate(cal) if other != day for step in range(92)]
        for block, step in enumerate(range(0, 95, 4)):
            ranked = sorted(eligible, key=lambda item: (float(np.square(
                (donors[item[0], item[1]] - donors[i, step]) / sd).sum()), item[0] * 95 + item[1]))
            expected = sorted(j * 95 + start for j, start in ranked[:32])
            np.testing.assert_array_equal(pools[i][block], expected)
    bank = np.load(parent / 'pit_bank.npz')['pit']
    noise, selected = paths(bank, cal, pools, return_donors=True)
    np.testing.assert_array_equal(noise, paths(bank, cal, pools))
    case = 0
    for member in range(5):
        for draw in range(8):
            for index, day in enumerate(cal):
                seed = int.from_bytes(hashlib.sha256(f'e66:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                rng = np.random.default_rng(seed)
                copied = []
                for block in range(24):
                    pool = pools[index][block]
                    chosen = int(pool[int(rng.integers(len(pool)))])
                    assert chosen == selected[case, block]
                    assert cal[chosen // 95] != day
                    copied.extend([bank[member, chosen + offset].copy() for offset in range(4)])
                np.testing.assert_array_equal(noise[case], np.array(copied[:95]))
                case += 1
    query = donors[:1].copy()
    original = donor_pools(query, donors, cal[:1], cal, mean, sd)
    changed = query.copy()
    changed[:, 8:] += 1e6
    future = donor_pools(changed, donors, cal[:1], cal, mean, sd)
    for block in (0, 1):
        np.testing.assert_array_equal(original[0][block], future[0][block])
    global_pool = donor_pools(query, donors, cal[:1], cal, mean, sd, k=None)
    full_pool = donor_pools(query, donors, cal[:1], cal, mean, sd, k=29 * 92)
    np.testing.assert_array_equal(paths(bank, cal[:1], global_pool), paths(bank, cal[:1], full_pool))
    flat = donor_pools(np.zeros((1, 95, 4)), np.zeros((30, 95, 4)), cal[:1], cal,
                       np.zeros(4), np.ones(4))
    first_eligible = np.array([j * 95 + step for j in range(1, 30) for step in range(92)])[:32]
    np.testing.assert_array_equal(flat[0][0], first_eligible)
    counts = np.bincount((selected // 95).reshape(-1), minlength=30)
    result = {'passed': True, 'calibration_only': True, 'validation_evaluated': False,
              'test_evaluated': False, 'new_model_fitting': False, 'candidate_quality_claim': False,
              'independent_all_calibration_pools_exact': True,
              'independent_all_calibration_paths_exact': True, 'path_count': case,
              'self_date_donors_excluded': True, 'no_cross_date_blocks': True,
              'earlier_blocks_future_exogenous_invariant': True, 'full_pool_identity_exact': True,
              'deterministic_tie_order_exact': True, 'repeated_paths_exact': True,
              'common_air_comparisons_share_paths_by_construction': True,
              'fit_context_mean': mean.tolist(), 'fit_context_sd': sd.tolist(),
              'fit_sd_floor_used': floor.tolist(), 'calibration_donor_counts': counts.tolist(),
              'data_sha256': digest(data_path), 'freeze_canonical_sha256': digest(args.freeze),
              'bank_array_sha256': hashlib.sha256(bank.tobytes()).hexdigest(),
              'fit_days': fit, 'calibration_days': cal,
              'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('EXOGENOUS_BLOCK_PREFLIGHT_COMPLETE', case, flush=True)


if __name__ == '__main__':
    main()
