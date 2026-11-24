"""Replace one e113 command at a time and close the loop with the frozen weighted policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e113'))
import run as e113
from darrow_env import action_to_air

ROOT = Path('/home/jio/fcsrl_src/results/e113/main')


def read(path):
    return json.loads(path.read_text())


def verify(path, digest):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path


def action(policy, obs):
    # Preserve the e113 evaluation's matrix shape also for small groups.
    assert 0 < len(obs) <= 200
    if len(obs) == 200:
        return policy.act_det(obs)[:, 0]
    pad = np.repeat(obs[:1], 200-len(obs), axis=0)
    return policy.act_det(np.concatenate([obs,pad]))[:len(obs), 0]


def load(seed, hashes):
    archive, agents = {}, {}
    for arm in ('uniform','weighted'):
        root = ROOT / arm / f'seed_{seed}'
        path = root / 'winter_rollout.npz'
        verify(path, hashes[str(path.relative_to(ROOT))])
        with np.load(path) as data:
            archive[arm] = {key: data[key] for key in ('states','air')}
        checkpoint = root/'model_final'
        for p in checkpoint.iterdir():
            verify(p,hashes[str(p.relative_to(ROOT))])
        agent = e113.base.common.make_agent('fcsrl_actor',seed,e113.prepared,e113.base.common.config(66))
        agent.load(checkpoint)
        agents[arm] = agent
    return archive, agents


def simulate(env, episodes, agents, native, ids, times, online=False):
    """Until replacement every step must agree with the archived weighted trajectory."""
    obs = env.reset(episodes)
    n = len(ids)
    changed = np.full(n,-1)
    trace = np.empty((n,95,4))  # air, NH4, max DO, executed minus native policy proposal
    previous = obs[:,0].copy()
    for t in range(95):
        orig = action(agents['weighted'], obs)
        prefix = changed < 0
        np.testing.assert_allclose(action_to_air(orig[prefix]),native['air'][ids[prefix],t],rtol=0,atol=.05)
        executed = orig.copy()
        if online or np.any(times==t):
            other = action(agents['uniform'],obs)
            if online:
                use = prefix & (t>0) & (obs[:,0]>=3) & (obs[:,0]>previous) & (action_to_air(other)-action_to_air(orig)>=250)
            else:
                use = times==t
            assert np.all(changed[use]==-1)
            executed[use] = other[use]
            changed[use] = t
        previous = obs[:,0].copy()
        obs, _, cost, done, info = env.step(executed)
        control = changed < 0
        np.testing.assert_allclose(env.state[control],native['states'][ids[control],t],atol=2e-6,rtol=0)
        nh = env.state[:,0]
        do = env.state[:,2:].max(1)
        np.testing.assert_array_equal(cost,(nh>4).astype(int)+(do>2))
        trace[:,t] = np.column_stack([info['air'],nh,do,action_to_air(executed)-action_to_air(orig)])
    assert done
    if not online:
        np.testing.assert_array_equal(changed,times)
    return trace, changed


def run(out, smoke):
    plan = read(ROOT/'plan.json')
    for name, value in plan['input_sha256'].items():
        verify(e113.prepared/name,value)
    hashes = read(ROOT/'checksums.json')
    dates = plan['winter']
    splits, days = e113.base.load_selected(e113.prepared/'data.pkl')
    assert splits['val'] == dates
    e113.base.DeviceConfig().select_device(-1)
    env = e113.BatchEnv(days,e113.FastDynamics(e113.prepared/'surrogate.pkl'))
    cases = pd.read_csv(HERE.parent/'e114/pairs.csv').query('opposite and gap<=1000')
    assert len(cases)==11
    cases = cases[['day','member','seed_hi','nh_hi']].drop_duplicates().rename(columns={'member':'origin','seed_hi':'seed','nh_hi':'native_gap'})
    out.mkdir(parents=True,exist_ok=False)
    cases.to_csv(out/'cases.csv',index=False)
    all_episodes = [(d,m) for d in dates for m in range(5)]
    all_ids = np.arange(200)
    for seed in ([int(cases.seed.iloc[0])] if smoke else range(5)):
        archives,agents = load(seed,hashes)
        uniform,weighted = (archives[k] for k in ('uniform','weighted'))
        for arm in ('uniform','weighted'):
            source = archives[arm]
            replay,_ = simulate(env,all_episodes,dict(weighted=agents[arm]),source,all_ids,np.full(200,-1))
            np.testing.assert_allclose(replay[:,:,0],source['air'],atol=.05,rtol=0)
            np.testing.assert_array_equal(replay[:,:,1]>4,source['states'][:,:,0]>4)
            np.testing.assert_array_equal(replay[:,:,2]>2,source['states'][:,:,2:].max(2)>2)
        if smoke:
            test_times=np.array([0,94,-1])
            trace,changed = simulate(env,[all_episodes[i] for i in (0,1,0)],agents,weighted,np.array([0,1,0]),test_times)
            for count, tt in enumerate(changed):
                mask = np.arange(95)!=tt
                assert np.all(trace[count,mask,3]==0)
            print('Full native replay, day boundaries, no-op verified.',flush=True)
            return
        trace,triggers = simulate(env,all_episodes,agents,weighted,all_ids,np.full(200,-1),online=True)
        np.savez_compressed(out/f'online{seed}.npz',trace=trace,trigger=triggers)
        tasks = []
        for case in cases[cases.seed==seed].itertuples():
            d = dates.index(case.day)
            i = d*5+case.origin
            gap = np.cumsum((weighted['states'][i,:,0]>4).astype(int)-(uniform['states'][i,:,0]>4))
            assert gap[-1]==case.native_gap and gap[-1]>0
            event = int(np.flatnonzero(gap>0)[0])
            for t in range(event+1):
                tasks.extend((d,case.origin,event,t,m) for m in range(5))
        parts = []
        for start in range(0,len(tasks),200):
            chunk = np.array(tasks[start:start+200])
            d,_,_,t,m = chunk.T
            ids = 5*d+m
            episodes = [all_episodes[i] for i in ids]
            trace,_ = simulate(env,episodes,agents,weighted,ids,t)
            parts.append(trace)
        if tasks:
            np.savez_compressed(out/f'branches{seed}.npz',keys=np.array(tasks),trace=np.concatenate(parts))
        print('seed',seed,'case/time/member branches', len(tasks),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(a.out,a.smoke)
