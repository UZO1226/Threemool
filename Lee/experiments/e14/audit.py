"""Reproducible environment comparison, without accessing test observations."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CODE = Path(__file__).resolve().parents[2] / 'code/darrow'
sys.path.insert(0, str(CODE))
from direct_dynamics import DirectDeltaRegressor, PersistenceRegressor
from surrogate_dw import STATE, EXO, ZONES, GBMDynamics, build_arrays, monotone_constraints


def measure(data, path, split='val'):
    table, days = data['data'], data['days'][split]
    dyn = GBMDynamics(path)
    x, delta = build_arrays(table, days)
    pred = np.mean([dyn.delta(x, np.full(len(x), m)) for m in range(dyn.n_members)], 0)
    true = x[:, :5] + delta
    train_states = np.concatenate([table.loc[d][STATE].to_numpy(float) for d in data['days']['train']])
    sd = train_states.std(0)
    rows = [table.loc[d] for d in days]
    initial = np.stack([g[STATE].to_numpy(float)[0] for g in rows])
    truth = np.stack([g[STATE].to_numpy(float)[1:] for g in rows])
    exo = np.stack([g[EXO].to_numpy(float) for g in rows])
    zones = np.stack([g[ZONES].to_numpy(float) for g in rows])
    trajectories = []
    for m in range(dyn.n_members):
        state = initial.copy()
        traj = []
        for t in range(95):
            state = dyn.step(state, exo[:, t], zones[:, t], zones[:, max(0,t-1)], np.full(len(rows), m))
            traj.append(state.copy())
        trajectories.append(np.stack(traj, 1))
    traj = np.stack(trajectories)
    mae = np.abs(traj-truth[None]).mean(axis=(0,1,2))
    result = {
        'delta_r2': dict(zip(STATE, (1-((pred-delta)**2).sum(0)/((delta-delta.mean(0))**2).sum(0)).tolist())),
        'next_state_r2': dict(zip(STATE, (1-((pred-delta)**2).sum(0)/((true-true.mean(0))**2).sum(0)).tolist())),
        'next_state_mae': dict(zip(STATE, np.abs(pred-delta).mean(0).tolist())),
        'rollout_mae': dict(zip(STATE, mae.tolist())),
        'normalized_rollout_mae': float((mae / sd).mean()),
        'nh4_violation_pred_actual_pct': [float((traj[...,0]>4).mean()*100),float((truth[...,0]>4).mean()*100)],
        'do_violation_pred_actual_pct': [float((traj[...,2:].max(-1)>2).mean()*100),float((truth[...,2:].max(-1)>2).mean()*100)],
    }
    base = x[:,:5]
    z = x[:,14:19]*1000
    zp = x[:,19:24]*1000
    sensitivity = []
    for m in range(dyn.n_members):
        low=dyn.step(base,x[:,5:14],z*.75,zp,np.full(len(x),m))
        high=dyn.step(base,x[:,5:14],z*1.25,zp,np.full(len(x),m))
        sensitivity.append((high-low).mean(0))
    result['mean_one_step_air_effect'] = dict(zip(STATE,np.mean(sensitivity,0).tolist()))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    data=pd.read_pickle(a.data)
    xt,yt=build_arrays(data['data'],data['days']['train'])
    states=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in data['days']['train']])
    hi=states.max(0)*1.1
    paths={'old':a.old}
    persist=a.out/'persistence.pkl'
    with persist.open('wb') as f: pickle.dump({'models':[[PersistenceRegressor() for _ in STATE]],'hi':hi},f)
    paths['persistence']=persist
    for leaves in (7,15):
        models=[]
        for m in range(5):
            idx=np.random.RandomState(m).choice(len(xt),int(.8*len(xt)),replace=False)
            row=[]
            for j in range(5):
                c=monotone_constraints(j)
                c[j]=1
                model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=leaves,
                    min_samples_leaf=40,l2_regularization=1,monotonic_cst=c,early_stopping=False,random_state=m+j)
                model.fit(xt[idx],(xt[:,:5]+yt)[idx,j])
                row.append(DirectDeltaRegressor(model,j))
            models.append(row)
        path=a.out/f'direct_{leaves}.pkl'
        with path.open('wb') as f: pickle.dump({'models':models,'hi':hi},f)
        paths[f'direct_{leaves}']=path
        print('TRAINED',leaves,flush=True)
    scores={}
    for name,path in paths.items():
        scores[name]=measure(data,path)
        print(name,json.dumps(scores[name]),flush=True)
    old=scores['old']
    eligible=[name for name in scores if name.startswith('direct') and
        scores[name]['normalized_rollout_mae'] <= .9*old['normalized_rollout_mae'] and
        scores[name]['rollout_mae']['nh4']<=old['rollout_mae']['nh4'] and
        all(scores[name]['rollout_mae'][k]<=1.1*old['rollout_mae'][k] for k in STATE[1:])]
    selected=min(eligible,key=lambda k:scores[k]['normalized_rollout_mae']) if eligible else 'old'
    report={'scores':scores,'selected':selected,'selected_path':str(paths[selected].resolve()),
        'test_accessed':False,'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
        'model_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}}
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('SELECTED',selected,flush=True)


if __name__=='__main__':
    main()
