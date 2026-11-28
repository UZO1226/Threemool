"""Full-day matched tests of a bounded intervention on frozen FCSRL."""
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
MODES = ('base', 'pulse', 'recover')
METRICS = ('air', 'nh', 'do', 'excess')


def correction(t, mode, obs, previous_nh, q, tree, support, pulse_time, cuts):
    if mode == 'base':
        return np.zeros_like(q)
    pulse = (t <= 70) & (pulse_time < 0) & (prior.command(obs, q, tree, support, 'both') == 1)
    delta = np.where(pulse, prior.STEP, 0.)
    if mode == 'recover':
        low, high = support[1][prior.state_grid(obs[:, 0], obs[:, 10]*9000, support[0])].T
        cut = ((pulse_time >= 0) & (t > pulse_time) & (t <= pulse_time + 24) & (cuts < 2)
               & (obs[:, 0] <= 4) & (obs[:, 0] <= previous_nh)
               & (q <= np.minimum(high, 9000)) & (q-250 >= np.maximum(low, 0)))
        delta[cut] = -250.
        cuts[cut] += 1
    pulse_time[pulse] = t
    return delta


def rollout(mode, agent, tree, support, env, dates):
    obs = env.reset(E.all_episodes(dates, 5))
    previous_nh = obs[:, 0].copy()
    pulse_time = np.full(len(obs), -1)
    cuts = np.zeros(len(obs), dtype=int)
    traces = {k: [] for k in ('states', 'air', 'raw', 'delta')}
    for t in range(95):
        action = agent.act_det(obs)[:, 0]
        q = prior.action_to_air(action)
        delta = correction(t, mode, obs, previous_nh, q, tree, support, pulse_time, cuts)
        previous_nh = obs[:, 0].copy()
        act = np.where(delta == 0, action, prior.air_to_action(q+delta))
        obs, _, cost, done, info = env.step(act)
        np.testing.assert_allclose(info['air'] - q, delta, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(cost, info['v_nh4'] + info['v_do'])
        for k,v in zip(traces, (env.state.copy(), info['air'], q, delta)):
            traces[k].append(v)
    assert done
    return {k: np.stack(v, axis=1) for k, v in traces.items()}


def audit(group, ref, dates):
    base, pulse, recover = (group[k] for k in MODES)
    with np.load(ref) as saved:
        np.testing.assert_array_equal(saved['dates'], np.asarray(dates, dtype=str))
        for k in ('air', 'states'):
            np.testing.assert_array_equal(base[k], saved[k])
    np.testing.assert_array_equal(pulse['delta'] > 0, recover['delta'] > 0)
    for mode, z in group.items():
        diff = z['delta']
        assert np.isin(diff, [-250,0,500]).all()
        ups, cuts = diff > 0, diff < 0
        assert not ups[:, 71:].any() and np.all(ups.sum(1) <= 1)
        assert np.all(cuts.sum(1) <= 2*ups.sum(1))
        if mode != 'recover':
            assert not cuts.any()
        ep,t = np.where(cuts)
        first = ups.argmax(1)[ep]
        assert np.all((t > first) & (t <= first + 24))
        if len(t):
            nh = z['states'][ep, t-1, 0].astype(np.float32)
            old = z['states'][ep, np.maximum(0,t-2), 0].astype(np.float32)
            assert (nh <= 4).all() and (nh[t>=2] <= old[t>=2]).all()
    for a,b,changed in [(base,pulse,pulse['delta']>0), (pulse,recover,recover['delta']<0)]:
        shared = changed.cumsum(1) == 0
        for k in ('air','states'):
            np.testing.assert_array_equal(a[k][shared], b[k][shared])


def statistics(z):
    state = z['states']
    nh = (state[:,:,0] > 4).sum(1)
    do = (state[:,:,2:].max(2) > 2).sum(1)
    ups, cuts = (z['delta']>0).sum(1), (z['delta']<0).sum(1)
    stats = dict(air=z['air'].sum(1)*.25, nh=nh, do=do,
                 excess=np.maximum(0, state[:,:,0]-4).sum(1)*.25,
                 limit_pct=100*((nh+do)<=12), raw_air=z['raw'].sum(1)*.25,
                 added=ups*125, removed=cuts*62.5)
    means = {k:float(v.mean()) for k,v in stats.items()}
    np.testing.assert_allclose(means['air'],means['raw_air']+means['added']-means['removed'],atol=1e-8,rtol=0)
    return dict(**means, pulsed=int(sum(ups)), cut_once_or_more=int(sum(cuts>=1)), cut_twice=int(sum(cuts==2)))


def smoke():
    class High:
        def predict(self, x):
            return np.ones(len(x))
    x = np.zeros((3,11))
    x[:,0] = [3,3,5]
    support = (np.array([[1,2,3],[0,1,2]]), np.tile([0,5000], (16,1)))
    q = np.full(3,3000.)
    p, c = np.full(3,-1), np.zeros(3,int)
    def apply(t, old):
        return correction(t,'recover',x,np.array(old),q,High(),support,p,c)
    for t, old, expect in [(0,[3,3,5],[500]*3),(1,[3,2,5],[-250,0,0]),
                           (2,[3,2,5],[-250,0,0]),(3,[3,2,5],[0,0,0]),
                           (24,[3,3,5],[0,-250,0]),(25,[3,3,5],[0,0,0])]:
        np.testing.assert_array_equal(apply(t,old), expect)
    print('smoke passed')


def main(source, out):
    torch.set_num_threads(1)
    hashes = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    tree_hashes = json.loads((HERE.parent/'e79/summary.json').read_text())['e78_input_sha256']
    splits, days = E.S.base.load_selected(E.DATA/'data.pkl')
    dates, rows = E.load_train()
    assert dates == splits['train'] and len(splits['val']) == 40
    support = prior.fit_support(rows[:140])
    env = E.BatchEnv(days, E.Dynamics(E.DATA/'surrogate.pkl'))
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        agent = E.restore(seed)
        fingerprint = E.S.fingerprint(agent)
        path = source/f'tree{seed}.joblib'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == tree_hashes[path.name]
        tree = joblib.load(path)
        group = {m:rollout(m,agent,tree,support,env,splits['val']) for m in MODES}
        audit(group, source/f'{seed}_winter_base.npz', splits['val'])
        for m,z in group.items():
            np.savez_compressed(out/f'{seed}_{m}.npz', dates=np.asarray(splits['val'],dtype=str), **z)
            records.append(dict(seed=seed, mode=m, **statistics(z)))
        assert E.S.fingerprint(agent) == fingerprint
        print('audited seed',seed,flush=True)
    (out/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/jio/fcsrl_src/results/e78/main'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.out:
        main(args.source,args.out)
    else:
        parser.error('--out required for full run')
