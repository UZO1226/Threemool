"""Aggregate archived raw traces and check the fixed single-step and winter criteria."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('/home/jio/fcsrl_src/results/e113/main')
NAMES = ['air','nh','do','excess']


def daily(air,nh,do):
    return np.stack([air.sum(1)*.25, (nh>4).sum(1), (do>2).sum(1), np.maximum(nh-4,0).sum(1)*.25],axis=1)


def archived(seed,arm):
    with np.load(ROOT/arm/f'seed_{seed}/winter_rollout.npz') as a:
        nh = a['states'][:,:,0]
        return daily(a['air'],nh,a['states'][:,:,2:].max(2)), nh


def compare(source,out):
    dates = json.loads((ROOT/'plan.json').read_text())['winter']
    cases = pd.read_csv(source/'cases.csv')
    online, branches, windows = [], [], []
    for seed in range(5):
        w,wn = archived(seed,'weighted')
        u,un = archived(seed,'uniform')
        with np.load(source/f'online{seed}.npz') as z:
            trace,trigger = z['trace'],z['trigger']
            assert np.isfinite(trace).all() and trace.shape==(200,95,4)
            value = daily(*trace[:,:,:3].transpose(2,0,1))
            change = value-w
            used = trigger>=0
            assert np.all((trigger[used]>=1)&(trigger[used]<95))
            assert (trace[np.flatnonzero(used),trigger[used],3]>=250).all()
            assert np.all(np.where(np.arange(95)[None,:]!=trigger[:,None],trace[:,:,3],0)==0)
            for i,t in zip(np.flatnonzero(used),trigger[used]):
                assert trace[i,t-1,1]>=3-1e-6
            for period,indices in [('all',np.arange(200)),('early',np.arange(100)),('late',np.arange(100,200))]:
                row=dict(seed=seed,period=period,episodes=len(indices),triggered=int(used[indices].sum()))
                for j,name in enumerate(NAMES):
                    row['weighted_'+name]=w[indices,j].mean()
                    row[name]=change[indices,j].mean()
                row['joint']=row['nh']<0 and row['air']<=.05 and row['do']<=0
                online.append(row)
        file=source/f'branches{seed}.npz'
        if not file.exists():
            assert (cases.seed!=seed).all()
            continue
        with np.load(file) as z:
            keys,trace=z['keys'],z['trace']
            assert np.isfinite(trace).all() and len(np.unique(keys,axis=0))==len(keys)
            value=daily(*trace[:,:,:3].transpose(2,0,1))
            ids=keys[:,0]*5+keys[:,4]
            change=value-w[ids]
            for idx,(d,orig,event,t,m) in enumerate(keys):
                native=5*d+m
                proposed=trace[idx,t,3]
                assert np.all(trace[idx,:t,3]==0) and np.all(trace[idx,t+1:,3]==0)
                np.testing.assert_array_equal(trace[idx,:t,1]>4,wn[native,:t]>4)
                row=dict(day=dates[d],period='early' if d<20 else 'late',seed=seed,origin=orig,member=m,
                         event=event,t=t,lead_min=(event+1-t)*15,command_delta=proposed,
                         native_gap=int(w[native,1]-u[native,1]),
                         event_nh_delta=int((trace[idx,:event+1,1]>4).sum()-(wn[native,:event+1]>4).sum()),
                         still_saves=float(u[native,0]-value[idx,0]))
                row.update(zip(NAMES,change[idx]))
                row['joint']=row['nh']<0 and row['do']<=0 and row['air']<=.05
                row['retains']=row['nh']<0 and row['do']<=0 and row['still_saves']>.05
                branches.append(row)
    online=pd.DataFrame(online)
    b=pd.DataFrame(branches)
    for (day,seed,orig),g in b.groupby(['day','seed','origin']):
        same=g[g.origin==g.member].sort_values('t')
        assert list(same.t)==list(range(same.event.iloc[0]+1)) and len(g)==5*len(same)
        cross=g.groupby('t').joint.sum()
        good=same[same.joint].sort_values(['nh','air','t'])
        date_id=dates.index(day)
        row=dict(day=day,period=same.period.iloc[0],seed=seed,origin=orig,event=int(same.event.iloc[0]),
                 native_gap=int(same.native_gap.iloc[0]),n=len(same),nh_better=int((same.nh<0).sum()),
                 joint=int(same.joint.sum()),keeps_savings=int(same.retains.sum()),repeat_t=int(((cross>=4)&same.set_index('t').joint).sum()),
                 best_t=-1,air=0.,nh=0.,do=0.,excess=0.,event_nh_delta=0,other_success=0,command_delta=0.)
        if len(good):
            best=good.iloc[0]
            row.update(best_t=int(best.t),other_success=int(cross[best.t])-1,
                       **{k:best[k] for k in [*NAMES,'event_nh_delta','command_delta']})
        row['recovered_fraction']=min(1.,-row['nh']/row['native_gap'])
        windows.append(row)
    results=pd.DataFrame(windows)
    assert set(map(tuple,results[['day','seed','origin','native_gap']].values))==set(map(tuple,cases[['day','seed','origin','native_gap']].values))
    out.mkdir(parents=True,exist_ok=True)
    b.to_csv(out/'branches.csv',index=False)
    results.to_csv(out/'cases.csv',index=False)
    online.to_csv(out/'online.csv',index=False)
    per_period={}
    for period in ('all','early','late'):
        c=results if period=='all' else results[results.period==period]
        r=online[online.period==period]
        avg=r[NAMES].mean()
        day_share=c.assign(repeated=c.repeat_t>0).groupby('day').repeated.mean()
        per_period[period]=dict(cases=len(c), candidates=int(c.n.sum()), joint_commands=int(c.joint.sum()),
            cases_nh=int((c.nh_better>0).sum()),cases_retains=int((c.keeps_savings>0).sum()), cases_joint=int((c.joint>0).sum()),
            cases_cross=int((c.repeat_t>0).sum()),days=len(day_share),days_cross=int((day_share>.5).sum()),
            oracle_mean_delta=c.groupby('day')[NAMES].mean().mean().to_dict(),
            trigger_episodes=int(r.triggered.sum()),online_joint=int(r.joint.sum()),online_mean_delta=avg.to_dict(),
            trigger_pass=bool(r.joint.sum()>=4 and avg.nh<0 and avg.air<=.05 and avg['do']<=0))
    outdata=dict(periods=per_period, repeatable=all(per_period[p]['days_cross']>per_period[p]['days']/2 for p in ('early','late')),
                 online_pass=all(per_period[p]['trigger_pass'] for p in ('early','late')))
    (out/'decision.json').write_text(json.dumps(outdata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(outdata,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    compare(a.source,a.out)
