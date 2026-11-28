"""Train one local action correction per frozen FCSRL seed and compare its execution modes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e77'))
import diagnose as E
from darrow_env import action_to_air, air_to_action

STEP = 500.
MODES = ('base', 'down', 'both')


def state_grid(nh4, prev, edges):
    return np.searchsorted(edges[0], nh4) * 4 + np.searchsorted(edges[1], prev)


def fit_support(rows):
    nh4 = np.concatenate([r['state'][1:95, 0] for r in rows])
    prev = np.concatenate([r['air'][:94] for r in rows])
    air = np.concatenate([r['air'][1:95] for r in rows])
    edges = np.quantile(np.stack([nh4, prev]), [.25, .5, .75], axis=1).T
    grid = state_grid(nh4, prev, edges)
    bounds = np.full((16, 2), np.nan)
    for i in range(16):
        sample = air[grid == i]
        if len(sample) >= 100:
            bounds[i] = np.quantile(sample, [.05, .95])
    return edges, bounds


def allowed(obs, q, support):
    edges, bounds = support
    low, high = bounds[state_grid(obs[:, 0], obs[:, 10] * 9000, edges)].T
    inside = (low <= q) & (q <= high)
    return inside & (q - STEP >= np.maximum(low, 0)), inside & (q + STEP <= np.minimum(high, 9000))


def observations(rows, keys):
    flat = []
    for d, t in keys:
        row = rows[d]
        q, temp, sin, cos, inf = row['exo'][t, :5]
        flat.append([*row['state'][t], q/1000, temp/10, sin, cos, inf/50, row['air'][t-1]/9000])
    return np.asarray(flat, dtype=np.float32)


def fit(seed, agent, rows, dyn, support, out):
    keys = np.array([(d,t) for d in range(140) for t in range(4,89,4)])
    x = observations(rows, keys)
    q = action_to_air(agent.act_det(x)[:,0])
    down, up = allowed(x, q, support)
    use = down | up
    keys, x, q, down, up = [a[use] for a in (keys, x, q, down, up)]
    if len(x) < 20:
        raise ValueError(f'Insufficient supported training rows: {len(x)}')
    parts = []
    for i in range(0, len(keys), 180):
        ix = slice(i, i+180)
        branches = []
        for delta in (-STEP, 0., STEP):
            obs, sums, _ = E.simulate(rows, dyn, keys[ix], np.clip(q[ix]+delta, 0, 9000), agent)
            np.testing.assert_array_equal(obs, x[ix])
            branches.append(sums)
        parts.append(branches)
    lo, mid, hi = [np.concatenate(v) for v in zip(*parts)]
    down_votes = E.classify(lo, mid)[0] == 1
    up_votes = E.classify(mid, hi)[0] == 2
    labels = np.where(down & (down_votes.sum(1) >= 4), -1,
                      np.where(up & (up_votes.sum(1) >= 4), 1, 0))
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight='balanced', random_state=77)
    tree.fit(x, labels)
    joblib.dump(tree, out/f'tree{seed}.joblib')
    np.savez_compressed(out/f'fit{seed}.npz', keys=keys, x=x, q=q, low=lo, base=mid, high=hi,
                        down=down, up=up, labels=labels)
    return tree, {str(v): int((labels == v).sum()) for v in (-1,0,1)}


def command(obs, q, tree, support, mode):
    if mode == 'base':
        return np.zeros(len(q), dtype=int)
    down, up = allowed(obs, q, support)
    label = tree.predict(obs)
    return np.where((label == -1) & down, -1,
                    np.where((mode == 'both') & (label == 1) & up, 1, 0))


def evaluate(seed, agent, tree, support, mode, split, dates, days, dyn, out):
    env = E.BatchEnv(days, dyn)
    obs = env.reset(E.all_episodes(dates,5))
    states, airs, changes, original = [], [], [], []
    for t in range(95):
        action = agent.act_det(obs)[:,0]
        q = action_to_air(action)
        direction = command(obs, q, tree, support, mode)
        if mode == 'down':
            assert np.all(direction <= 0)
        act = np.where(direction == 0, action, air_to_action(q + direction * STEP))
        obs, reward, cost, done, info = env.step(act)
        np.testing.assert_allclose(info['air'] - q, direction * STEP, atol=1e-8, rtol=0)
        assert np.allclose(reward, -info['air']/9000) and np.array_equal(cost,info['v_nh4']+info['v_do'])
        states.append(env.state.copy())
        airs.append(info['air'])
        original.append(q)
        changes.append(direction)
    assert done
    state, air, change, native = [np.stack(a, axis=1) for a in (states, airs, changes, original)]
    if mode == 'base':
        ref = 'val' if split == 'winter' else 'train'
        with np.load(E.POLICIES/f'seed_{seed}'/f'{ref}_rollout.npz') as z:
            for name, arr in [('air', air), ('states', state)]:
                expected = z[name] if ref == 'val' else z[name][-len(arr):]
                np.testing.assert_allclose(arr, expected, atol=2e-4, rtol=1e-7)
    nh = (state[:,:,0] > 4).sum(1)
    do = (state[:,:,2:].max(2) > 2).sum(1)
    per_day = dict(air=air.sum(1)*.25, nh=nh, do=do, excess=np.maximum(state[:,:,0]-4,0).sum(1)*.25,
                   limit_pct=100*((nh+do)<=12), reward=-air.sum(1)/9000,
                   up_steps=(change==1).sum(1), down_steps=(change==-1).sum(1),
                   raw_air=native.sum(1)*.25)
    np.savez_compressed(out/f'{seed}_{split}_{mode}.npz', states=state, air=air, direction=change, raw=native,
                        dates=np.asarray(dates,dtype=str), **{'day_'+k:v for k,v in per_day.items()})
    return {k:float(v.mean()) for k,v in per_day.items()}


def main(out, smoke):
    torch.set_num_threads(1)
    dates, rows = E.load_train()
    dyn = E.Dynamics(E.DATA/'surrogate.pkl')
    support = fit_support(rows[:140])
    if smoke:
        agent = E.restore(0)
        keys = [(0,4),(1,12)]
        x = observations(rows,keys)
        q = action_to_air(agent.act_det(x)[:,0])
        obs, sums, _ = E.simulate(rows,dyn,keys,q,agent)
        np.testing.assert_array_equal(obs,x)
        repeated = E.simulate(rows,dyn,keys,q,agent)[1]
        np.testing.assert_array_equal(sums,repeated)
        assert np.all(E.classify(sums,repeated)[0] == 0)
        class Forced:
            def predict(self, obs):
                return np.array([-1,1,1])
        test_obs = np.repeat(x[:1], 3, axis=0)
        test_support = (support[0], np.tile([1000, 5000], (16,1)))
        for mode, expected in zip(MODES, [[0,0,0],[-1,0,0],[-1,1,0]]):
            np.testing.assert_array_equal(command(test_obs,np.array([3000,3000,4900]),Forced(),test_support,mode),expected)
        print('smoke passed')
        return
    out.mkdir(parents=True, exist_ok=False)
    splits, days = E.S.base.load_selected(E.DATA/'data.pkl')
    assert dates == list(splits['train']) and not set(dates) & set(splits['val'])
    hashes = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    records = []
    for seed in range(5):
        agent = E.restore(seed)
        before = E.S.fingerprint(agent)
        tree, counts = fit(seed, agent, rows, dyn, support, out)
        for split, group in [('train48', dates[140:]), ('winter', splits['val'])]:
            for mode in MODES:
                stats = evaluate(seed,agent,tree,support,mode,split,group,days,dyn,out)
                records.append(dict(seed=seed, mode=mode, split=split, fit_labels=counts, **stats))
        assert E.S.fingerprint(agent) == before
        print('complete',seed,counts,flush=True)
    (out/'records.json').write_text(json.dumps(records,indent=1,allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    main(args.out,args.smoke)
