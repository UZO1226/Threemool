"""Compare an upward flow slew limit to the unchanged e81 recovery and FCSRL controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e81'))
import recover as R

E = R.E
ARCHIVE = Path('/home/jio/fcsrl_src/results/e81/main')
TREES = Path('/home/jio/fcsrl_src/results/e78/main')


def apply_limit(k, explicit, proposal, previous, nh, old_nh):
    window = (k>=1) & (k<=4) & (explicit==0)
    excess = np.maximum(0, proposal-previous-125*k)
    risk = (nh>=4) | (2*nh-old_nh>=4)
    requested = window & (excess>1e-8)
    withheld = np.where(requested & ~risk,excess,0.)
    return withheld, requested & risk


def rollout(env, dates, agent, tree, support):
    obs = env.reset(E.all_episodes(dates,5))
    n = len(obs)
    old_nh = obs[:,0].copy()
    pulse, cuts, last = np.full(n,-1), np.zeros(n,int), np.full(n,-100)
    trace = {k:[] for k in ('states','air','raw','delta','withheld','bypassed')}
    for t in range(95):
        action = agent.act_det(obs)[:,0]
        q = R.prior.action_to_air(action)
        explicit = R.correction(t,'recover',obs,old_nh,q,tree,support,pulse,cuts)
        withheld,bypassed = apply_limit(t-last,explicit,q+explicit,env.prev_air,obs[:,0],old_nh)
        delta = explicit-withheld
        act = np.where(delta==0,action,R.prior.air_to_action(q+delta))
        last[explicit<0] = t
        old_nh = obs[:,0].copy()
        obs,_,cost,done,info = env.step(act)
        np.testing.assert_allclose(info['air']-q,delta,atol=1e-8,rtol=0)
        np.testing.assert_array_equal(cost,info['v_nh4']+info['v_do'])
        for key,v in zip(trace,(env.state.copy(),info['air'],q,explicit,withheld,bypassed)):
            trace[key].append(v)
    assert done
    return {k:np.stack(v,axis=1) for k,v in trace.items()}


def smoke():
    age=np.array([0,1,2,3,4,5,1,1,1,1,1])
    proposal=np.array([4000]*10+[2800])
    explicit=np.zeros(11)
    explicit[6]=-250
    nh=np.array([3.]*11)
    nh[7:10]=[4.,3.6,3.5]
    previous_nh=np.full(11,3.5)
    previous_nh[8]=3.
    withheld,bypass=apply_limit(age,explicit,proposal,3000,nh,previous_nh)
    np.testing.assert_array_equal(withheld,[0,875,750,625,500,0,0,0,0,875,0])
    np.testing.assert_array_equal(np.flatnonzero(bypass),[7,8])
    print('slew and bypass checks passed')


def metrics(z):
    s = z['states']
    nh=(s[:,:,0]>4).sum(1)
    do=(s[:,:,2:].max(2)>2).sum(1)
    return np.column_stack([z['air'].sum(1)*.25,nh,do,np.maximum(s[:,:,0]-4,0).sum(1)*.25,
                            100*((nh+do)<=12)])


def audit(z,control):
    w=z['withheld']
    shared=(w>0).cumsum(1)==0
    for k in ['air','states','delta','raw']:
        np.testing.assert_array_equal(z[k][shared],control[k][shared])
    np.testing.assert_allclose(z['air']-z['raw'],z['delta']-w,atol=1e-8,rtol=0)
    assert not np.any((w>0)&z['bypassed'])
    for ep,t in zip(*np.where((w>0)|z['bypassed'])):
        last=np.flatnonzero(z['delta'][ep,:t]<0)[-1]
        assert 1<=t-last<=4 and z['delta'][ep,t]==0
        nh,old=z['states'][ep,[t-1,t-2],0].astype(np.float32)
        risk=nh>=4 or 2*float(nh)-old>=4
        assert risk==z['bypassed'][ep,t]
        if not risk:
            np.testing.assert_allclose(z['air'][ep,t]-z['air'][ep,t-1],125*(t-last),atol=1e-8,rtol=0)


def main(out):
    torch.set_num_threads(1)
    hashes=json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    tree_hashes=json.loads((HERE.parent/'e79/summary.json').read_text())['e78_input_sha256']
    splits,days=E.S.base.load_selected(E.DATA/'data.pkl')
    dates,rows=E.load_train()
    assert dates==splits['train'] and len(splits['val'])==40
    support=R.prior.fit_support(rows[:140])
    env=E.BatchEnv(days,E.Dynamics(E.DATA/'surrogate.pkl'))
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for seed in range(5):
        agent=E.restore(seed)
        before=E.S.fingerprint(agent)
        path=TREES/f'tree{seed}.joblib'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==tree_hashes[path.name]
        tree=joblib.load(path)
        for mode in ['base','recover','limited']:
            if mode=='limited':
                z=rollout(env,splits['val'],agent,tree,support)
                audit(z,control)
                more=dict(clipped_steps=int((z['withheld']>0).sum()),clipped_episodes=int((z['withheld']>0).any(1).sum()),
                          bypassed_steps=int(z['bypassed'].sum()),withheld_volume=float(z['withheld'].sum(1).mean()*.25))
                np.savez_compressed(out/f'{seed}_limited.npz',dates=np.array(splits['val'],str),**z)
            else:
                z=R.rollout(mode,agent,tree,support,env,splits['val'])
                with np.load(ARCHIVE/f'{seed}_{mode}.npz') as ref:
                    np.testing.assert_array_equal(ref['dates'],np.array(splits['val'],str))
                    for k in z:
                        np.testing.assert_array_equal(z[k],ref[k])
                if mode=='recover':
                    control=z
                more={}
            records.append(dict(seed=seed,mode=mode,**dict(zip(['air','nh','do','excess','limit_pct'],metrics(z).mean(0))),**more))
        assert E.S.fingerprint(agent)==before
        print('audited seed',seed,flush=True)
    (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:
        smoke()
    elif a.out:
        main(a.out)
    else:
        p.error('--out is required')
