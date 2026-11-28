"""Constrained choice between two frozen hour-one FCSRL continuations."""
import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e92'))
import bounds

SCOPES = ('pooled', 'date', 'window')
FAMILIES = {'fixed': [1, 2], 'optional': [0, 1, 2]}


def candidates(a, b, c, first, eligible):
    no_second = np.where(first[:, None], b, a)
    with_second = np.where((first & eligible)[:, None], c, no_second)
    return np.stack([a, no_second, with_second], axis=1)


def smoke():
    bounds.smoke()
    a = np.zeros((4, 4))
    a[:, 0] = 100
    b = a + [[-10, 1, 0, 0], [-9, 2, 1, 0], [-3, -1, 0, 0], [-8, 1, 0, 0]]
    c = a + [[-2, 0, 0, 0], [-1, -1, 0, 0], [-4, 0, 1, 0], [-5, -1, 0, 0]]
    table = candidates(a, b, c, np.array([0, 1, 1, 1], bool), np.array([1, 0, 1, 1], bool))
    np.testing.assert_array_equal(table[0], [a[0]] * 3)
    np.testing.assert_array_equal(table[1, 1:], [b[1]] * 2)
    day = np.array([1, 1, 1, 2])
    for modes in FAMILIES.values():
        for scope in SCOPES:
            feasible = []
            for idx in itertools.product(modes, repeat=len(day)):
                diff = table[np.arange(len(day)), idx] - a
                dates = bounds.daily(diff, day)
                metric = {'pooled': dates.mean(0)[None], 'date': dates, 'window': diff}[scope]
                if (metric[:, 1:] <= 0).all():
                    feasible.append(dates[:, 0].mean())
            sol = bounds.evaluate(table, day, modes, scope)
            if feasible:
                np.testing.assert_allclose(sol['air_bound'], min(feasible), atol=1e-8)
            else:
                assert sol['status'] == 'infeasible'


def examine(day, table, first, possible, rule, ref):
    count = len(day)
    rows = np.arange(count)
    delta = table - table[:, :1]
    assert (count, first.sum(), possible.sum()) == (ref['n'], ref['first'], ref['possible'])
    random = np.zeros(count)
    for d in np.unique(day):
        ix = (day == d) & possible
        if ix.any():
            random[ix] = rule[ix].mean()
    effect = delta[:, 2] - delta[:, 1]
    perfect = possible & (effect[:, 0] < -bounds.TOL) & (effect[:, 1:] <= bounds.TOL).all(1)
    for key, take in [('first_only', np.zeros(count)), ('all_possible', possible), ('nh_rule', rule),
                      ('matched_random', random), ('perfect', perfect)]:
        outcome = bounds.daily(delta[:, 1] + take[:, None] * effect, day).mean(0)
        np.testing.assert_allclose(outcome, ref['modes'][key]['vs_base'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(take.sum(), ref['modes'][key]['second'])
    bad = (delta[:, 1:, 1:] > bounds.TOL).any(axis=2).all(axis=1)
    assert not (bad & ~first).any()
    result = dict(n=count, days=len(set(day)), first=int(first.sum()), possible=int(possible.sum()),
                  bad_rows=rows[bad].tolist(), bad_no_second=int((bad & ~possible).sum()),
                  baseline_air=float(bounds.daily(table[:, 0, 0], day).mean()))
    for family, modes in FAMILIES.items():
        solutions = {}
        for scope in SCOPES:
            sol = bounds.evaluate(table, day, modes, scope)
            if sol['status'] == 'optimal':
                idx = np.array(sol['selected'])
                raw_delta = table[rows, idx] - table[:, 0]
                w = 1 / np.bincount(day)[day]
                np.testing.assert_allclose(np.average(raw_delta, weights=w, axis=0), sol['delta'], atol=1e-8, rtol=0)
                sol['changed_first'] = int(((idx == 0) & first).sum())
                sol['second'] = int(((idx == 2) & possible).sum())
            elif scope == 'window' and family == 'fixed':
                np.testing.assert_array_equal(sol['failed_units'], rows[bad])
            solutions[scope] = sol
        result[family] = solutions
    return result


def run(out):
    ref = json.loads((HERE.parent / 'e94/summary.json').read_text())
    records = []
    for seed in range(5):
        path = bounds.ROOT / 'e94/main' / f'pulse{seed}.npz'
        with np.load(path) as z:
            keys, first, eligible = z['keys'], z['first'], z['eligible']
            rule, supported = z['rule'], z['supported']
            assert not (rule & ~eligible).any() and len(np.unique(keys, axis=0)) == len(keys)
            table = candidates(z['a'], z['b'], z['c'], first, eligible)
        assert np.isfinite(table).all()
        groups = {}
        for name, mask in [('supported', supported), ('all', np.ones(len(keys), bool))]:
            g = examine(keys[mask, 0], table[mask], first[mask], (eligible & first)[mask],
                        (rule & first)[mask], ref[seed]['groups'][name])
            groups[name] = g
            print(seed, name, g['fixed']['pooled']['status'], flush=True)
        records.append(dict(seed=seed, sha256=sha256(path.read_bytes()).hexdigest(), groups=groups))
    with out.open('x') as f:
        json.dump(records, f, indent=1, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    smoke()
    if args.out:
        run(args.out)
