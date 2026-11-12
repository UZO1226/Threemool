"""Train-only block residual simulation and proper distribution scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
from cooperative import arrays,metric,gate
from surrogate_dw import GBMDynamics,STATE,build_arrays,features


def bank(dyn,data,days):
    x,delta=build_arrays(data['data'],days)
    rows=[]
    for m in range(dyn.n_members):
        residual=delta-dyn.delta(x,np.full(len(x),m))
        residual-=residual.mean(0)
        residual[:,1]=0
        rows.append(residual)
    return np.stack(rows)


def noise_paths(residual,days,draws=8):
    out=[]
    for m in range(len(residual)):
        for draw in range(draws):
            for day in days:
                seed=int.from_bytes(hashlib.sha256(f'20261112:{day}:{m}:{draw}'.encode()).digest()[:4],'little')
                rng=np.random.default_rng(seed)
                start=rng.integers(0,len(residual[m])//95,size=24)*95+rng.integers(0,92,size=24)
                indices=(start[:,None]+np.arange(4)).reshape(-1)[:95]
                out.append(residual[m,indices])
    return np.stack(out)


def rollout(dyn,batch,noise,scale,draws=8,air_scale=1.,initial_delta=0.):
    true,exo,zone=batch
    n=len(true)
    members=np.repeat(np.arange(dyn.n_members),draws*n)
    state=np.tile(true[:,0],(dyn.n_members*draws,1))+initial_delta
    exo=np.tile(exo,(dyn.n_members*draws,1,1))
    zone=zone*air_scale
    zone*=np.minimum(1.,9000/np.maximum(zone.sum(-1,keepdims=True),1))
    zone=np.tile(zone,(dyn.n_members*draws,1,1))
    trajectory=[]
    for t in range(95):
        x=features(state,exo[:,t],zone[:,t],zone[:,max(0,t-1)])
        state=np.clip(state+dyn.delta(x,members)+noise[:,t]*scale,0,dyn.hi)
        trajectory.append(state.copy())
    return np.stack(trajectory,1).reshape(dyn.n_members,draws,n,95,5)


def crps(samples,true):
    n=len(samples)
    first=np.abs(samples-true).mean(0)
    weights=(2*np.arange(1,n+1)-n-1).reshape((n,)+(1,)*(samples.ndim-1))
    second=(weights*np.sort(samples,axis=0)).sum(0)/(n*n)
    return first-second


def summarize(traj,true,sd):
    m,r,n,t,j=traj.shape
    samples=traj.reshape(m*r,n,t,j)
    v=metric(samples,true,sd)
    target=true[:,1:]
    v['nh4_violation']['member_pred']=((traj[...,0]>4).mean((1,2,3))*100).tolist()
    v['do_violation']['member_pred']=((traj[...,2:].max(-1)>2).mean((1,2,3))*100).tolist()
    concentration_crps=crps(samples,target).mean((0,1))
    pn=(samples[...,0]>4).mean(0);pd_=(samples[...,2:].max(-1)>2).mean(0)
    bn=float(((pn-(target[...,0]>4))**2).mean())
    bd=float(((pd_-(target[...,2:].max(-1)>2))**2).mean())
    v['crps']=dict(zip(STATE,concentration_crps.tolist()))
    v['normalized_crps']=float((concentration_crps/sd).mean())
    v['nh4_brier']=bn;v['do_brier']=bd
    v['proper_selection_score']=v['normalized_crps']+bn+bd
    v['point_mean_mae']=dict(zip(STATE,np.abs(samples.mean(0)-target).mean((0,1)).tolist()))
    v['structural_members']=m;v['noise_draws_per_member']=r
    return v


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--contractive-summary',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.environ['DARROW_FAST_DYNAMICS']='1'
    data=pd.read_pickle(a.data)
    source=json.loads(a.contractive_summary.read_text())
    selected=source['selected_by_inner_calibration']
    path=Path(source['model_paths'][selected]);dyn=GBMDynamics(path)
    fit,cal=source['fit_days'],source['calibration_days']
    sd=np.concatenate([data['data'].loc[day][STATE].to_numpy(float) for day in fit]).std(0)
    residual=bank(dyn,data,cal)
    np.savez_compressed(a.out/'residual_bank.npz',residual=residual)
    calibration_batch=arrays(data['data'],cal)
    noise=noise_paths(residual,cal)
    # Independent exact scoring identity check, including the single-sample case.
    tiny=np.array([[0.,2.],[1.,4.],[3.,5.]])
    target=np.array([2.,3.])
    pair=.5*np.abs(tiny[:,None]-tiny[None,:]).mean((0,1))
    np.testing.assert_allclose(crps(tiny,target),np.abs(tiny-target).mean(0)-pair)
    np.testing.assert_allclose(crps(tiny[:1],target),np.abs(tiny[0]-target))
    results={};scales={}
    for sn in [0,.25,.5,1]:
        for sd_ in [0,.5,1,1.5]:
            key=f'n{sn}_d{sd_}'
            scale=np.array([sn,0,sd_,sd_,sd_])
            traj=rollout(dyn,calibration_batch,noise,scale)
            results[key]=summarize(traj,calibration_batch[0],sd)
            scales[key]=scale
            print('INNER',key,results[key]['proper_selection_score'],flush=True)
    winner=min(results,key=lambda k:results[k]['proper_selection_score'])
    days=data['days']['val'];batch=arrays(data['data'],days)
    nval=noise_paths(residual,days)
    scores={}
    for key in dict.fromkeys(['n0_d0',winner]):
        traj=rollout(dyn,batch,nval,scales[key])
        scores[key]=summarize(traj,batch[0],sd)
        low=rollout(dyn,batch,nval,scales[key],air_scale=.75)
        high=rollout(dyn,batch,nval,scales[key],air_scale=1.25)
        scores[key]['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
            'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
    selected_traj=rollout(dyn,batch,nval,scales[winner])
    perturbed=rollout(dyn,batch,nval,scales[winner],initial_delta=.01)
    coefficients=np.array([model.state_coefficient for model in dyn.models[0]])
    bound=.01*coefficients[None,None,None,None,:]**np.arange(1,96)[None,None,None,:,None]
    assert np.all(np.abs(perturbed-selected_traj)<=bound+1e-9)
    sensitivity=summarize(rollout(dyn,batch,noise_paths(residual,days,draws=16),scales[winner],draws=16),batch[0],sd)
    reference=source['validation']['old']
    deterministic=source['validation'][selected]
    for key in ['normalized_mae']:
        np.testing.assert_allclose(scores['n0_d0'][key],deterministic[key],rtol=1e-10,atol=1e-10)
    result={'selected_scale':winner,'scale_vector':scales[winner].tolist(),'base_model':selected,
        'base_model_path':str(path.resolve()),'base_model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'residual_bank_sha256':hashlib.sha256((a.out/'residual_bank.npz').read_bytes()).hexdigest(),
        'gate':gate(scores[winner],reference),'inner':results,'validation':scores,'test_evaluated':False,
        'verification':{'crps_pair_identity':True,'zero_noise_matches_deterministic':True,'contraction_with_common_innovations':True},
        'sixteen_noise_draw_sensitivity':sensitivity,
        'code_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',winner,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
