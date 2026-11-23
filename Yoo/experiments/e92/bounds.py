"""Perfect-information air minimization over stored six-hour FCSRL branches."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e84'))
from choice import solve, TOL

ROOT = Path('/home/jio/fcsrl_src/results')
MODES = ('A','B','D')


def weights(day):
    _,ix,n = np.unique(day,return_inverse=True,return_counts=True)
    return 1/n[ix]


def daily(values,day):
    return np.stack([values[day==d].mean(axis=0) for d in np.unique(day)])


def pooled(delta,day):
    w = weights(day)
    scaled = delta * (w/w.mean())[:,None,None]
    floors = np.average(delta[:,:,1:].min(axis=1),weights=w,axis=0)
    if (floors>TOL).any():
        return None,dict(status='infeasible',method='single_metric_lower_bound',floors=floors.tolist())
    take, info = solve(scaled,False)
    return take,dict(**info,floors=floors.tolist())


def evaluate(totals, day, modes, scope):
    n=len(day)
    delta=totals[:,modes] - totals[:,:1]
    dates=np.unique(day)
    if scope=='pooled':
        ids,meta=pooled(delta,day)
    elif scope=='date':
        ids=np.empty(n,dtype=int)
        failed=[]
        proofs=[]
        for date in dates:
            ix=day==date
            selected,proof=pooled(delta[ix],day[ix])
            proofs.append(proof)
            if selected is None:
                failed.append(int(date))
            else:
                ids[ix]=selected
        if failed:
            return dict(status='infeasible',failed_dates=failed,date_proofs=proofs)
        meta=dict(status='optimal',air_bound=float(np.mean([p['air_bound'] for p in proofs])),date_proofs=proofs)
    else:
        ids,meta=solve(delta,True)
        if ids is None:
            meta['failed_dates']=np.unique(day[meta['failed_units']]).tolist()
        else:
            # e84's enumeration bound uses equal rows; reweight its per-row optima by date.
            meta['air_bound']=float(np.average(delta[np.arange(n),ids,0],weights=weights(day)))
    if ids is None:
        return meta
    branches=np.array(modes)[ids]
    change=totals[np.arange(n),branches]-totals[:,0]
    daily_delta=daily(change,day)
    mean=daily_delta.mean(0)
    np.testing.assert_allclose(mean[0],meta['air_bound'],rtol=0,atol=1e-5)
    assert np.all(mean[1:]<=TOL)
    worsened=(daily_delta[:,1:]>TOL).any(axis=1)
    if scope!='pooled':
        assert not worsened.any()
    if scope=='window':
        assert np.all(change[:,1:]<=TOL)
    return dict(**meta,delta=mean.tolist(),savings_pct=float(-100*mean[0]/daily(totals[:,0,0],day).mean()),
                worse_days=int(worsened.sum()),worse_windows=int((change[:,1:]>TOL).any(axis=1).sum()),
                counts=np.bincount(branches,minlength=3).tolist(),selected=branches.tolist())


def smoke():
    # Unequal daily counts distinguish mean-by-date from equal-row optimization.
    table=np.zeros((4,3,4))
    table[:,1]=[[-20,1,0,0],[-16,1,0,0],[-8,1,0,0],[12,-1,0,0]]
    table[:,2]=[[-2,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]
    table[:,:,0]+=1000
    day=np.array([0,0,0,1])
    for scope in ('pooled','date','window'):
        for modes in [(1,2),(0,1,2)]:
            opt=[]
            for choice in itertools.product(modes,repeat=4):
                dx=table[np.arange(4),choice]-table[:,0]
                by_day=daily(dx,day)
                quality={'pooled':by_day.mean(0)[None,:],'date':by_day,'window':dx}[scope]
                if (quality[:,1:]<=0).all():
                    opt.append(by_day.mean(0)[0])
            sol=evaluate(table,day,modes,scope)
            np.testing.assert_allclose(sol['delta'][0],min(opt),atol=TOL)
    table[:,1:,3]+=1
    assert all(evaluate(table,day,[1,2],s)['status']=='infeasible' for s in ('pooled','date','window'))


def run(root,out):
    previous=json.loads((HERE.parent/'e91/summary.json').read_text())
    result=[]
    for r in previous:
        seed=r['seed']
        path=root/'e91/main'/f'hour{seed}.npz'
        with np.load(path) as z:
            keys, totals, supported, keep=z['keys'],z['totals'],z['supported'],z['keep'][1]
        assert np.isfinite(totals).all() and np.unique(keys,axis=0).shape==keys.shape
        row=dict(seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups={})
        for group,mask in [('supported',supported),('all',np.ones(len(keys),bool))]:
            subset=totals[mask]
            day=keys[mask,0]
            ref=r['groups'][group]
            assert ref['n']==len(day) and ref['days']==len(np.unique(day))
            a,b,d=subset.transpose(1,0,2)
            selected=np.where(keep[mask,None],b,d)
            for arm,old in [(selected,ref['trend']['outcome']), (b,ref['always']), (d,ref['remove'])]:
                np.testing.assert_allclose(daily(arm-a,day).mean(0),old['vs_base'],rtol=0,atol=1e-8)
            stats=dict(n=len(day),days=ref['days'],baseline_air=float(daily(a[:,0],day).mean()),
                       e91_trend=ref['trend']['outcome']['vs_base'])
            for family,modes in [('fixed',[1,2]), ('optional',[0,1,2])]:
                stats[family]={s:evaluate(subset,day,modes,s) for s in ('pooled','date','window')}
                print(seed,group,family,stats[family]['pooled']['status'],
                      stats[family]['pooled'].get('air_bound'),flush=True)
            row['groups'][group]=stats
        result.append(row)
    with out.open('x') as f:
        json.dump(result,f,indent=1,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke:
        smoke()
    else:
        if args.out is None:
            p.error('supply --out')
        run(args.root,args.out)
