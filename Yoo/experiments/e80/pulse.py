"""Matched single-action rollouts on measured and previously visited DARROW states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e78'))
import run as prior

E = prior.E
TIMES = np.arange(4, 69, 4)
HORIZONS = (4, 12, 24)
METRICS = ['air', 'nh', 'do', 'excess']


def measured(rows, agent, tree, support):
    keys = np.array([(d, t) for d in range(len(rows)) for t in TIMES])
    obs = prior.observations(rows, keys)
    q = prior.action_to_air(agent.act_det(obs)[:, 0])
    take = prior.command(obs, q, tree, support, 'both') == 1
    keys = keys[take]
    return np.column_stack((np.repeat(keys, 5, axis=0), np.tile(np.arange(5), len(keys)))), np.repeat(q[take], 5)


def simulated(archive):
    ep, col = np.where(archive['direction'][:, TIMES] == 1)
    t = TIMES[col]
    return np.column_stack((ep // 5, t, ep % 5)), archive['raw'][ep, t]


def branch(rows, dyn, agent, keys, q, delta, archive=None):
    windows = {i: {k: a[t:t+25] for k, a in rows[d].items()} for i, (d, t, _) in enumerate(keys)}
    env = E.BatchEnv(windows, dyn)
    env.reset([(i, m) for i, (_, _, m) in enumerate(keys)])
    if archive is None:
        env.prev_air = np.array([rows[d]['air'][t-1] for d, t, _ in keys])
        env.prev_zone = np.array([rows[d]['zone'][t-1] for d, t, _ in keys])
    else:
        d, t, m = keys.T
        ep = d * 5 + m
        env.state = archive['states'][ep, t-1].copy()
        env.prev_air = archive['air'][ep, t-1].copy()
        env.prev_zone = env.prev_air[:, None] * np.array([rows[day]['frac'][step-1] for day, step in zip(d, t)])
    obs = env.obs()
    np.testing.assert_allclose(prior.action_to_air(agent.act_det(obs)[:, 0]), q, atol=.02, rtol=0)
    totals = []
    for i in range(24):
        act = prior.air_to_action(q + delta) if i == 0 else agent.act_det(obs)[:, 0]
        obs, _, cost, done, info = env.step(act)
        if i == 0:
            np.testing.assert_allclose(info['air'], q + delta, atol=1e-8, rtol=0)
            if archive is not None and delta == prior.STEP:
                np.testing.assert_allclose(env.state, archive['states'][ep, t], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(cost, info['v_nh4'] + info['v_do'])
        totals.append(np.column_stack((info['air']*.25, info['v_nh4'], info['v_do'],
                                       np.maximum(env.state[:, 0] - 4, 0)*.25)))
    assert done
    return np.stack(totals, axis=1).cumsum(axis=1)[:, np.array(HORIZONS)-1]


def summarize(keys, diffs):
    dates = np.unique(keys[:, 0])
    result = dict(days=len(dates), pairs=len(keys))
    if not len(dates):
        return result
    daily = np.stack([diffs[keys[:, 0] == d].mean(0) for d in dates])
    result['means'] = daily.mean(0).tolist()
    late = daily[:, -1] - daily[:, 0]
    result['late_mean'] = late.mean(0).tolist()
    result['late_negative_days'] = (late < -1e-8).sum(0).tolist()
    result['both_nh_negative_days'] = int(((late[:, 1] < -1e-8) & (late[:, 3] < -1e-8)).sum())
    return result


def main(out, source, smoke):
    torch.set_num_threads(1)
    hashes = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    train, rows = E.load_train()
    splits, days = E.S.base.load_selected(E.DATA/'data.pkl')
    assert train == splits['train']
    dyn = E.Dynamics(E.DATA/'surrogate.pkl')
    if smoke:
        agent = E.restore(1)
        anchors = np.array([[0, 4], [1, 12]])
        keys = np.column_stack((np.repeat(anchors, 5, axis=0), np.tile(np.arange(5), 2)))
        q = prior.action_to_air(agent.act_det(prior.observations(rows, anchors))[:, 0])
        for delta in (0, prior.STEP):
            old = E.simulate(rows, dyn, anchors, q+delta, agent)[1]
            new = branch(rows, dyn, agent, keys, q.repeat(5), delta)
            np.testing.assert_allclose(new[:, 0], old.reshape(-1, 4), atol=1e-8, rtol=0)
        print('four-step smoke passed')
        return
    out.mkdir(parents=True, exist_ok=False)
    support = prior.fit_support(rows[:140])
    winter = [days[d] for d in splits['val']]
    result = dict(metrics=METRICS, steps=HORIZONS, rows=[])
    tree_hashes = json.loads((HERE.parent/'e79/summary.json').read_text())['e78_input_sha256']
    for seed in range(5):
        agent = E.restore(seed)
        tree_file = source/f'tree{seed}.joblib'
        assert hashlib.sha256(tree_file.read_bytes()).hexdigest() == tree_hashes[tree_file.name]
        tree = joblib.load(tree_file)
        with np.load(source/f'{seed}_winter_both.npz') as saved:
            archive = {k:saved[k] for k in ['dates','direction','states','raw','air']}
        np.testing.assert_array_equal(archive['dates'], np.array(splits['val'], dtype=str))
        for group, data, dates, trajectory in [
            ('train48_measured', rows[140:], train[140:], None),
            ('winter_measured', winter, splits['val'], None),
            ('winter_policy', winter, splits['val'], archive),
        ]:
            if trajectory is None:
                keys, q = measured(data, agent, tree, support)
            else:
                keys, q = simulated(trajectory)
            totals = np.empty((len(keys), 2, len(HORIZONS), len(METRICS)))
            for start in range(0, len(keys), 250):
                b = slice(start, start + 250)
                for i, delta in enumerate((0, prior.STEP)):
                    totals[b, i] = branch(data, dyn, agent, keys[b], q[b], delta, trajectory)
            diffs = totals[:, 1] - totals[:, 0]
            assert np.isfinite(diffs).all()
            stats = summarize(keys, diffs)
            np.savez_compressed(out/f'{seed}_{group}.npz', dates=np.asarray(dates, dtype=str), keys=keys,
                                q=q, totals=totals)
            result['rows'].append(dict(seed=seed, group=group, **stats))
            print(seed, group, len(keys), stats['days'], flush=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/jio/fcsrl_src/results/e78/main'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    main(args.out, args.source, args.smoke)
