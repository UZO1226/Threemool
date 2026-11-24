"""Paired contrasts for two frozen e50 policies, from unshielded e59 trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jio/fcsrl_src/results')
COLS = ['air', 'cost', 'nh4', 'do']
EPS = 1e-8


def flags(x):
    primary = (x[...,0] < -EPS) & (x[...,1] < -EPS)
    water = (x[...,0] < -EPS) & (x[...,2] < -EPS) & (x[...,3] <= EPS)
    return primary, water


def stats(x):
    """x has shape (seed, date, metric), with five members already averaged."""
    p, w = flags(x)
    ps, ws = flags(x.mean(1))
    pdays, wdays = flags(x.mean(0))
    return dict(delta=dict(zip(COLS,x.mean((0,1)))), days=x.shape[1],
                joint_seed_dates=int(p.sum()), water_seed_dates=int(w.sum()),
                joint_seeds=int(ps.sum()), water_seeds=int(ws.sum()),
                joint_mean_days=int(pdays.sum()), water_mean_days=int(wdays.sum()),
                joint_4seed_days=int((p.sum(0)>=4).sum()), water_4seed_days=int((w.sum(0)>=4).sum()))


def episodes(method,seed,dates,ref):
    path=ROOT/f'e59/main/runs/none/{method}_s{seed}'
    meta=json.loads((path/'result.json').read_text())
    assert (meta['policy'],meta['seed'],meta['shield'])==(method,seed,'none')
    assert meta['reproduced_recorded_metrics'] and meta['checkpoint']==str(ROOT/f'e50/main/runs/{method}/seed_{seed}/model_final')
    rows=pd.read_json(path/'episodes.json',convert_dates=False)
    keys=list(zip(rows.day,rows.member))
    assert keys==[(d,m) for d in dates for m in range(5)]
    with np.load(path/'rollout.npz') as z:
        air,x=z['air'],z['states']
        assert air.shape==(200,95) and x.shape==(200,95,5)
        assert np.isfinite(x).all() and np.isfinite(air).all()
        volume=air.sum(1)*.25
        nh=(x[:,:,0]>4).sum(1)
        do=(x[:,:,2:].max(2)>2).sum(1)
    np.testing.assert_allclose(volume,rows.air_m3,rtol=0,atol=1e-8)
    np.testing.assert_array_equal(nh,rows.nh4_cost)
    np.testing.assert_array_equal(do,rows.do_cost)
    np.testing.assert_allclose(volume.mean(),ref['air_m3'],rtol=1e-7)
    np.testing.assert_allclose([(nh+do).mean(),nh.mean()*100/95,do.mean()*100/95],
                               [ref[k] for k in ('cost','nh4_viol_pct','do_viol_pct')],rtol=0,atol=1e-10)
    return np.stack([volume,nh+do,nh,do],axis=-1).reshape(40,5,len(COLS))


def smoke():
    x=np.array([[-1,-1,-1,0],[-1,0,-1,1],[0,-1,-1,0],[-1,-1,0,-1]])
    p,w=flags(x)
    np.testing.assert_array_equal(p,[1,0,0,1])
    np.testing.assert_array_equal(w,[1,0,0,0])


def run(out):
    data=ROOT/'e50/prepared/data.pkl'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==json.loads((HERE.parent/'e77/summary.json').read_text())['sha256'][str(data)]
    raw=pd.read_pickle(data)
    dates=list(raw['days']['val'])
    assert dates==sorted(dates) and len(dates)==40
    ref=json.loads((HERE.parent/'e50/summary.json').read_text())['per_seed']
    methods={m:np.stack([episodes(m,s,dates,ref[m][str(s)]) for s in range(5)]) for m in ['fcsrl','fcsrl_actor']}
    x=(methods['fcsrl_actor']-methods['fcsrl']).mean(axis=2)
    np.testing.assert_allclose(x[:,:,1],x[:,:,2]+x[:,:,3],atol=1e-12)
    cond=pd.DataFrame([raw['data'].loc[d][['q_in','temp']].mean() for d in dates],index=dates)
    groups={'early':np.arange(40)<20,'late':np.arange(40)>=20}
    for col in cond:
        low=cond[col].to_numpy()<=cond[col].median()
        groups[col+'_low']=low
        groups[col+'_high']=~low
    rng=np.random.default_rng(111)
    seeds=rng.integers(5,size=(5000,5))
    starts=rng.integers(40,size=(5000,6))
    days=((starts[:,:,None]+np.arange(7))%40).reshape(5000,-1)[:,:40]
    boot=x[seeds[:,:,None],days[:,None,:]].mean(axis=(1,2))
    s=dict(all=stats(x),interval=dict(zip(COLS,np.quantile(boot,[.025,.975],axis=0).T.tolist())),
           medians=cond.median().to_dict(),groups={k:stats(x[:,mask]) for k,mask in groups.items()},
           seeds=[stats(x[i:i+1]) for i in range(5)],
           without_seed=[stats(np.delete(x,i,axis=0))['delta'] for i in range(5)])
    out.mkdir(parents=True,exist_ok=False)
    index=pd.MultiIndex.from_product([range(5),dates],names=['seed','date'])
    pairs=pd.DataFrame(x.reshape(-1,4),index=index,columns=['delta_'+c for c in COLS])
    for m,v in methods.items():
        for j,c in enumerate(COLS):
            pairs[m+'_'+c]=v.mean(axis=2)[:,:,j].ravel()
    pairs['joint'],pairs['water_joint']=flags(x.reshape(-1,4))
    pairs.join(cond,on='date').to_csv(out/'pairs.csv')
    daily=pd.DataFrame(x.mean(0),index=pd.Index(dates,name='date'),columns=['delta_'+c for c in COLS])
    daily['joint_seeds'],daily['water_joint_seeds']=[v.sum(0) for v in flags(x)]
    daily.join(cond).to_csv(out/'dates.csv')
    (out/'summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    smoke()
    if args.out:
        run(args.out)
