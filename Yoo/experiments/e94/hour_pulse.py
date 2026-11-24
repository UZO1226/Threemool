"""Pair two one-hour decisions and let frozen FCSRL control the remaining steps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e90'))
import rebound as R

S = R.S
DOSE = 250.
HOUR = 4


class HourPulse:
    """simulate supplies the first action, so act_det starts at step one."""
    def __init__(self, agent, initial_nh, support):
        self.agent = agent
        self.initial_nh = initial_nh.astype(np.float32)
        self.support = support
        self.step = 0

    def act_det(self, obs):
        self.step += 1
        action = self.agent.act_det(obs)
        if self.step != HOUR:
            return action
        q = S.R.action_to_air(action[:, 0])
        edges, bounds = self.support
        grid = S.R.state_grid(obs[:, 0], obs[:, 10] * 9000, edges)
        low, high = bounds[grid].T
        self.eligible = (q <= np.minimum(high, 9000)) & (q - DOSE >= np.maximum(low, 0))
        self.level = obs[:, 0].copy()
        self.trend = self.level - self.initial_nh
        self.rule = self.eligible & (self.level <= 4) & (self.trend <= 0)
        return np.where(self.eligible, S.R.air_to_action(q - DOSE), action[:, 0])[:, None]


def branches(rows, archive, keys, q, agent, dyn, support):
    day, time, member = keys.T
    wrapped = HourPulse(agent, archive['states'][day * 5 + member, time - 1, 0], support)
    args = (rows, archive, keys, q)
    b, commands = R.simulate(*args, agent, dyn, True)
    c, modified = R.simulate(*args, wrapped, dyn, True)
    assert wrapped.step == R.STEPS - 1
    np.testing.assert_array_equal(commands[:, :HOUR], modified[:, :HOUR])
    np.testing.assert_array_equal(b[:, :HOUR], c[:, :HOUR])
    np.testing.assert_allclose(b[:, HOUR, 0] - c[:, HOUR, 0], .25 * DOSE * wrapped.eligible,
                               atol=1e-8, rtol=0)
    np.testing.assert_array_equal(b[~wrapped.eligible], c[~wrapped.eligible])
    return b, c, wrapped


def compare(days, first, eligible, rule, a, b, c):
    potential = first & eligible
    chosen = first & rule
    coin = np.zeros(len(days))
    for day in np.unique(days):
        group = (days == day) & potential
        if group.any():
            coin[group] = chosen[group].mean()
    np.testing.assert_allclose(np.bincount(days, weights=coin), np.bincount(days, weights=chosen))
    delta = c - b
    first_only = first[:, None] * (b - a)
    oracle = potential & (delta[:, 0] < -R.TOL) & (delta[:, 1:] <= R.TOL).all(1)
    w = S.weights(days)
    result = {}
    for name, take in [('first_only', np.zeros(len(days))), ('all_possible', potential),
                       ('nh_rule', chosen), ('matched_random', coin), ('perfect', oracle)]:
        extra = take[:, None] * delta
        result[name] = dict(second=float(take.sum()), vs_first=np.average(extra, axis=0, weights=w).tolist(),
                            vs_base=np.average(first_only + extra, axis=0, weights=w).tolist())
    return dict(n=len(days), dates=len(set(days)), first=int(first.sum()), possible=int(potential.sum()), modes=result)


def reference(experiment):
    return json.loads((HERE.parent / experiment / 'summary.json').read_text())


def run(out, smoke):
    torch.set_num_threads(1)
    assert all(S.digest(Path(p)) == v for p, v in reference('e77')['sha256'].items())
    dates, rows = S.E.load_train()
    support = S.R.fit_support(rows[:140])
    dyn = S.E.Dynamics(S.E.DATA / 'surrogate.pkl')
    root = S.ROOT
    if not smoke:
        out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        agent = S.E.restore(seed)
        before = S.E.S.fingerprint(agent)
        source = root / 'e78/main' / f'{seed}_train48_down.npz'
        probe = root / 'e85/main' / f'probe{seed}.npz'
        landmark = root / 'e91/main' / f'hour{seed}.npz'
        gate = root / 'e93/main' / f'gate{seed}.npz'
        assert S.digest(source) == reference('e89')[seed]['input_sha256'][1]
        assert S.digest(probe) == reference('e86')[seed]['score_sha256']
        assert S.digest(landmark) == reference('e92')[seed]['sha256']
        archive, sparse = S.read_candidates(source, dates[140:])
        with np.load(probe) as z:
            keys, q, expected = z['keys'], z['q'], z['totals']
        subset = S.headroom.nonoverlap(keys)
        np.testing.assert_array_equal(keys[subset], sparse)
        parts = []
        for start in range(0, len(keys), 250):
            sl = slice(start, start + 250)
            b, c, wrap = branches(rows[140:], archive, keys[sl], q[sl], agent, dyn, support)
            np.testing.assert_allclose(b.cumsum(1)[:, [3, 11, 23]], expected[sl, 1], atol=1e-7, rtol=0)
            if smoke:
                print('common prefix, direct dose and archived reference reproduced')
                return
            parts.append((c.sum(1), wrap.eligible, wrap.rule, wrap.level, wrap.trend))
        c, eligible, rule, level, trend = (np.concatenate(col)[subset] for col in zip(*parts))
        with np.load(landmark) as prior, np.load(gate) as frozen:
            for source_keys in (prior['keys'], frozen['keys']):
                np.testing.assert_array_equal(sparse, source_keys)
            np.testing.assert_allclose(level, prior['features'][:, 0], atol=1e-6, rtol=0)
            np.testing.assert_allclose(trend, prior['features'][:, 1], atol=2e-6, rtol=0)
            first = ~frozen['skip']
            supported = prior['supported']
            np.testing.assert_array_equal(supported, frozen['supported'])
            a, b = prior['totals'][:, :2].transpose(1, 0, 2)
            np.testing.assert_allclose(b, expected[subset, 1, -1], atol=1e-7, rtol=0)
        groups = {}
        for name, mask in [('supported', supported), ('all', np.ones(len(first), bool))]:
            g = compare(sparse[mask, 0], first[mask], eligible[mask], rule[mask], a[mask], b[mask], c[mask])
            ref = reference('e93')[seed]['groups'][name]
            assert g['first'] == ref['cut']
            np.testing.assert_allclose(g['modes']['first_only']['vs_base'], ref['delta']['gate'],
                                       rtol=0, atol=1e-7)
            groups[name] = g
        assert before == S.E.S.fingerprint(agent)
        np.savez_compressed(out / f'pulse{seed}.npz', keys=sparse, a=a, b=b, c=c, first=first,
                            supported=supported, eligible=eligible, rule=rule, level=level, trend=trend)
        records.append(dict(seed=seed, gate_sha256=S.digest(gate), groups=groups))
        print(seed, groups['supported']['modes']['nh_rule'], flush=True)
    (out / 'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.out, args.smoke)
