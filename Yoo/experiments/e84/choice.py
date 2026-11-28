"""Perfect-information selection among complete frozen FCSRL trajectories."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jio/fcsrl_src/results')
MODES = ('base','pulse','recover','limited')
TOL = 1e-8


def measure(z):
    states = z['states']
    assert states.shape == (200,95,5) and z['air'].shape == (200,95)
    return np.column_stack([z['air'].sum(1)*.25, (states[:,:,0]>4).sum(1),
                            (states[:,:,2:].max(2)>2).sum(1),
                            np.maximum(states[:,:,0]-4,0).sum(1)*.25])


def load(seed,root):
    cases, hashes = [], {}
    dates = None
    files = {}
    for m in MODES:
        exp = 'e83' if m == 'limited' else 'e81'
        path = root/exp/'main'/f'{seed}_{m}.npz'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            if dates is None:
                dates=z['dates'].copy()
                assert len(set(dates))==40
            np.testing.assert_array_equal(dates,z['dates'])
            cases.append(measure(z))
            files[m] = {k:z[k] for k in ('states','air','delta')}
        archive = json.loads((HERE.parent/exp/'records.json').read_text())
        r=next(r for r in archive if r['mode']==m and r['seed']==seed)
        np.testing.assert_allclose(cases[-1].mean(0),[r[k] for k in ('air','nh','do','excess')],rtol=0,atol=TOL)
    # Confirm a complete archived choice can be made at the claimed decision point.
    for a,b,change in [('base','pulse',files['pulse']['delta']>0),
                       ('pulse','recover',files['recover']['delta']<0)]:
        same_prefix=change.cumsum(1)==0
        for k in ('states','air'):
            np.testing.assert_array_equal(files[a][k][same_prefix],files[b][k][same_prefix])
    with np.load(root/'e83/main'/f'{seed}_limited.npz') as z:
        prefix=(z['withheld']>0).cumsum(1)==0
        for k in ('states','air'):
            np.testing.assert_array_equal(z[k][prefix],files['recover'][k][prefix])
    return dates,np.stack(cases,axis=1),hashes


def solve(delta,local):
    """delta is (selection unit, candidate, metric) relative to that unit's base."""
    n,k,_=delta.shape
    if local:
        valid=(delta[:,:,1:]<=0).all(2)
        failed=np.flatnonzero(~valid.any(1))
        if len(failed):
            return None,dict(status='infeasible',failed_units=failed.tolist())
        take=np.where(valid,delta[:,:,0],np.inf).argmin(1)
        return take,dict(status='optimal',method='enumeration',air_bound=float(delta[np.arange(n),take,0].mean()))
    # Remove exact duplicate choices for an individual unit to avoid symmetric binaries.
    rows=[]
    cols=[]
    for i,row in enumerate(delta):
        keep=np.unique(row,axis=0,return_index=True)[1]
        rows.extend([i]*len(keep))
        cols.extend(sorted(keep))
    rows,cols=np.asarray(rows),np.asarray(cols)
    options=delta[rows,cols]
    choices=csc_matrix((np.ones(len(rows)),(rows,np.arange(len(rows)))),shape=(n,len(rows)))
    constraints=LinearConstraint(vstack([choices,csc_matrix(options[:,1:].T)],format='csc'),
                                 np.r_[np.ones(n),[-np.inf]*3],np.r_[np.ones(n),np.zeros(3)])
    result=milp(options[:,0],integrality=np.ones(len(rows)),bounds=Bounds(0,1),constraints=constraints,
                options={'mip_rel_gap':0.0})
    if result.status==2:
        return None,dict(status='infeasible',method='milp')
    if not result.success:
        raise RuntimeError(result.message)
    x=result.x.round().astype(bool)
    np.testing.assert_allclose(x.astype(float),result.x,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(rows[x],np.arange(n))
    take=cols[x]
    objective=delta[np.arange(n),take,0].mean()
    bound=result.mip_dual_bound/n
    np.testing.assert_allclose(objective,bound,atol=1e-5,rtol=0)
    np.testing.assert_allclose(objective,result.fun/n,atol=1e-5,rtol=0)
    assert (delta[np.arange(n),take,1:].mean(0)<=TOL).all()
    return take,dict(status='optimal',method='milp',air_bound=float(bound),
                     gap_m3=float(abs(bound-objective)))


def smoke():
    fixture=np.array([[[0,0,0,0],[-5,1,0,0]],[[0,0,0,0],[1,-1,0,0]]],float)
    for z in [fixture,fixture+np.array([0,3,0,0]),fixture[:,1:]]:
        for local in [True,False]:
            opt=[]
            for path in itertools.product(range(z.shape[1]),repeat=len(z)):
                v=z[np.arange(len(z)),path]
                quality=v[:,1:] if local else v[:,1:].mean(0)
                if np.all(quality<=0):
                    opt.append(v[:,0].mean())
            ids,meta=solve(z,local)
            if opt:
                np.testing.assert_allclose(meta['air_bound'],min(opt),atol=TOL)
            else:
                assert ids is None and meta['status']=='infeasible'
    print('enumeration, cross-day compensation and infeasibility checks passed')


def evaluate(root):
    records=[]
    hashes={}
    reference_dates=None
    fixed=[]
    for seed in range(5):
        dates,v,h=load(seed,root)
        if reference_dates is None:
            reference_dates=dates
        np.testing.assert_array_equal(reference_dates,dates)
        hashes.update(h)
        fixed.append(v.mean(0).tolist())
        for family,offset in [('after_pulse',1),('skip_allowed',0)]:
            delta=v[:,offset:]-v[:,:1]
            for scope in ('episode','date','date_strict'):
                groups=delta if scope=='episode' else delta.reshape(40,5,*delta.shape[1:]).mean(1)
                ids,meta=solve(groups,scope=='date_strict')
                r=dict(seed=seed,family=family,scope=scope,**meta)
                if ids is not None:
                    ids=ids+offset
                    modes=ids if scope=='episode' else np.repeat(ids,5)
                    diff=v[np.arange(200),modes]-v[:,0]
                    day=diff.reshape(40,5,4).mean(1)
                    avg=day.mean(0)
                    assert np.all(avg[1:]<=TOL)
                    np.testing.assert_allclose(avg[0],meta['air_bound'],rtol=0,atol=1e-5)
                    bad=(day[:,1:]>TOL).any(1)
                    if scope=='date_strict':
                        assert not bad.any()
                    r.update(selected=ids.tolist(),delta=avg.tolist(),quality_worse_days=int(bad.sum()),
                             saving_pct=float(-100*avg[0]/v[:,0,0].mean()))
                records.append(r)
        print('solved seed',seed,flush=True)
    for seed in range(5):
        by={(r['family'],r['scope']):r.get('air_bound',np.inf) for r in records if r['seed']==seed}
        for fam in ['after_pulse','skip_allowed']:
            assert by[fam,'episode'] <= by[fam,'date']+1e-5 <= by[fam,'date_strict']+2e-5
        for scope in ['episode','date','date_strict']:
            assert by['skip_allowed',scope] <= min(0.,by['after_pulse',scope])+1e-5
    return dict(dates=reference_dates.tolist(),modes=MODES,metrics=['air','nh','do','excess'],
                fixed=fixed,records=records,source_sha256=hashes)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke:
        smoke()
    else:
        result=evaluate(args.root)
        with args.out.open('x') as f:
            json.dump(result,f,indent=1,allow_nan=False)
            f.write('\n')
