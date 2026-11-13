"""Independently check target sums, reset boundaries and the actual saved streams."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from targets import cost_rates, public_features


def explicit(cost, done, horizon):
    result = np.empty_like(cost, dtype=float)
    count = np.empty_like(cost, dtype=int)
    for t in range(len(cost)):
        for lane in range(cost.shape[1]):
            values = []
            for future in range(t, min(t + horizon, len(cost))):
                values.append(cost[future, lane])
                if done[future, lane]:
                    break
            result[t, lane] = np.mean(values)
            count[t, lane] = len(values)
    return result, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(56)
    cost = rng.integers(0, 2, size=(37, 3))
    done = rng.random(cost.shape) < .2
    for horizon in (1, 4):
        actual, n = cost_rates(cost, done, horizon)
        expected, counts = explicit(cost, done, horizon)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(n, counts)
    changed = cost.copy()
    done[8, :] = True
    changed[9:, :] = 1 - changed[9:, :]
    np.testing.assert_array_equal(cost_rates(cost, done, 4)[0][:9], cost_rates(changed, done, 4)[0][:9])
    for constant in (0, 1):
        result, _ = cost_rates(np.full(cost.shape, constant), done, 4)
        np.testing.assert_array_equal(result, np.full(cost.shape, constant))
    rows = []
    for seed in range(5):
        run = args.study / 'policies/new_ppolag_fcsrl' / f'seed_{seed}'
        fit = json.loads((run / 'correction/fit_summary.json').read_text())
        assert not fit['test_evaluated'] and fit['base_training_steps'] == 401280
        checked = 0
        for update in range(1, 67):
            path = run / 'transitions' / f'update_{update:03d}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == fit['capture_file_sha256'][path.name]
            with np.load(path, allow_pickle=False) as arrays:
                done = arrays['done']
                assert done.shape == (760, 8) and done[-1].all()
                for label in ('nh4', 'do'):
                    value = arrays[label]
                    one, _ = cost_rates(value, done, 1)
                    np.testing.assert_array_equal(one, value)
                    four, count = cost_rates(value, done, 4)
                    np.testing.assert_array_equal(four[done], value[done])
                    assert np.all(count[done] == 1)
                    if update in (1, 52, 53, 66):
                        expected, counts = explicit(value, done, 4)
                        np.testing.assert_array_equal(four, expected)
                        np.testing.assert_array_equal(count, counts)
                obs = arrays['obs'].reshape(-1, 11)
                action = arrays['action'].reshape(-1)
                x = public_features(obs, action)
                np.testing.assert_array_equal(x[:, :11], obs)
                np.testing.assert_array_equal(x[:, 11], action)
                np.testing.assert_array_equal(x[:, 1], obs[:, 1])
                checked += len(obs)
        assert checked == 401280
        rows.append({'policy_seed': seed, 'transitions_checked': checked,
                     'fit_updates': 52, 'held_updates': 14,
                     'independent_explicit_updates': [1, 52, 53, 66]})
    result = {'passed': True, 'policy_seeds': rows, 'test_evaluated': False,
              'horizon1_exact_cost_identity': True, 'horizon4_explicit_sums_exact': True,
              'terminal_partial_windows_correct': True, 'reset_future_perturbation_excluded': True,
              'constant_targets_correct': True, 'public_current_features_exact_no3_retained': True,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                for name in ('targets.py', 'check_targets.py')},
              'scope': 'Target construction on the original captured training streams only. No regressor fitting, validation/controller outcome or claim of horizon improvement.'}
    args.out.write_text(json.dumps(result, indent=2))
    print('E56_TARGET_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
