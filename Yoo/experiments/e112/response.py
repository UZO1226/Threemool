"""Time-lock archived control trajectories to NH4 rises and probe identical observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e59'))
import shield_eval as S
from darrow_env import STEP_H, AIR_MAX, action_to_air

ROOT=S.PREPARED.parents[1]
METRICS=['lag_min','miss','air_before','first_delta','hour_delta','rising','violations']


def onsets(nh):
    slope=np.diff(nh)
    out=[]
    for t in range(4,len(nh)-4):
        if slope[t-1]>.1 and slope[t-2]<=.1 and (not out or t>=out[-1]+4):
            out.append(t)
    return out


def records(nh,air,anchors,closed):
    rows=[]
    for t in anchors:
        before=air[t-1]
        delta=air[t:t+4]-before
        hits=np.flatnonzero(delta>=250.)
        rows.append(dict(t=t,lag_min=int(hits[0]*15) if len(hits) else 60,miss=int(not len(hits)),
                         air_before=before,first_delta=delta[0],hour_delta=delta.mean(),
                         rising=int(nh[t]-nh[t-1]>.1 and nh[t-1]-nh[t-2]<=.1),
                         violations=int(sum(nh[t+1:t+5]>4)) if closed else np.nan,
                         precut=bool(before<=air[t-4]-250)))
    return rows


def smoke():
    nh=np.array([0.,0.,0.,0.,1.,2.,2.,2.,3.,3.,3.,3.,3.])
    assert onsets(nh)==[4,8]
    air=np.full(12,3000.)
    for pos,lag in [(None,60),(4,0),(7,45)]:
        v=air.copy()
        if pos is not None:
            v[pos]+=250
        assert records(nh,v,[4],True)[0]['lag_min']==lag


def query(agent,state,exo,previous):
    obs=np.column_stack([state,exo[:,0]/1000,exo[:,1]/10,exo[:,2:4],exo[:,4]/50,previous/AIR_MAX])
    return action_to_air(agent.act_det(obs.astype(np.float32))[:,0])


def run(out):
    torch.set_num_threads(1)
    S.base.DeviceConfig().select_device(-1)
    data=S.PREPARED/'data.pkl'
    ref=json.loads((HERE.parent/'e77/summary.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest()==ref['sha256'][str(data)]
    splits,days=S.base.load_selected(data)
    val=splits['val']
    dates=splits['train'][:140]+val
    assert len(dates)==180
    halves={d:('train' if i<140 else 'early' if i<160 else 'late') for i,d in enumerate(dates)}
    anchors={d:onsets(days[d]['state'][:,0]) for d in dates}
    raw_state=np.concatenate([days[d]['state'][:-1] for d in dates])
    exo=np.concatenate([days[d]['exo'][:-1] for d in dates])
    previous=np.concatenate([days[d]['air'][np.maximum(0,np.arange(95)-1)] for d in dates])
    reference=pd.read_csv(HERE.parent/'e111/pairs.csv').set_index(['seed','date'])
    rows=[]
    def append(kind,day,member,nh,air,events,closed):
        rows.extend(dict(kind=kind,period=halves[day],day=day,member=member,method=method,seed=seed,**r)
                    for r in records(nh,air,events,closed))
    for method in ['fcsrl','fcsrl_actor']:
        for seed in range(5):
            agent,_,_=S.load_policy(method,seed,S.base.common.config(66))
            path=ROOT/f'e59/main/runs/none/{method}_s{seed}'
            info=json.loads((path/'result.json').read_text())
            assert S.base.fingerprint(agent)==info['policy_fingerprint']
            shared=query(agent,raw_state,exo,previous).reshape(len(dates),95)
            for d,air in zip(dates,shared):
                append('same_input',d,0,days[d]['state'][:,0],air,anchors[d],False)
            with np.load(path/'rollout.npz') as z:
                states,airs=z['states'],z['air']
            keys=[(r['day'],r['member']) for r in json.loads((path/'episodes.json').read_text())]
            assert keys==[(d,m) for d in val for m in range(5)]
            for j,d in enumerate(val):
                sl=slice(j*5,j*5+5)
                daily_air=airs[sl].sum(1).mean()*STEP_H
                daily_cost=((states[sl,:,0]>4).sum(1)+(states[sl,:,2:].max(-1)>2).sum(1)).mean()
                np.testing.assert_allclose([daily_air,daily_cost],
                    reference.loc[(seed,d),[method+'_air',method+'_cost']].to_numpy(float),rtol=0,atol=1e-8)
            for (d,member),state,air in zip(keys,states,airs):
                obs_state=np.vstack([days[d]['state'][0],state])
                nh=obs_state[:,0]
                # Check ordering, scales and action indexing against the native trajectory.
                reconstructed=query(agent,obs_state[:-1],days[d]['exo'][:-1],np.r_[days[d]['air'][0],air[:-1]])
                np.testing.assert_allclose(reconstructed,air,rtol=0,atol=.05)
                append('own',d,member,nh,air,onsets(nh),True)
                append('same_time',d,member,nh,air,anchors[d],True)
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['kind','day','member','method','seed','t']).any()
    # Counts for the shared anchors are identical for each seed and model.
    for k,mult in [('same_time',5),('same_input',1)]:
        for d,n in frame[frame.kind==k].groupby('day').size().items():
            assert n==len(anchors[d])*mult*10
    keys=['kind','period','method','seed']
    summaries=[]
    for subset,df in [('all',frame),('precut',frame[frame.precut])]:
        group=keys+['day','member']
        member=df.groupby(group)[METRICS].mean()
        daily=member.groupby(level=keys+['day']).mean()
        table=daily.groupby(level=keys).mean()
        table['days']=daily.groupby(level=keys).size()
        table['events']=df.groupby(keys).size()
        table=table.reindex(frame.groupby(keys).size().index)
        table[['days','events']]=table[['days','events']].fillna(0).astype(int)
        summaries.append(table.reset_index().assign(subset=subset))
    out.mkdir(parents=True,exist_ok=False)
    pd.concat(summaries).to_csv(out/'summary.csv',index=False)
    frame.to_csv(out/'events.csv.gz',index=False,compression={'method':'gzip','mtime':0})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    smoke()
    if args.out:
        run(args.out)
