"""Conditional stochastic transitions, with signs shared by all quantiles."""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e18'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e26'))
from cooperative import arrays,gate
from innovations import summarize
from correlated import event_diagnostics
from surrogate_dw import STATE,build_arrays,features,GBMDynamics
from direct_dynamics import DirectDeltaRegressor

LEVELS=np.array([.05,.25,.5,.75,.95])


def interpolate(values,u):
    lower=np.clip(np.searchsorted(LEVELS,u,side='right')-1,0,len(LEVELS)-2)
    low=np.take_along_axis(values,lower[None],axis=0)[0]
    high=np.take_along_axis(values,(lower+1)[None],axis=0)[0]
    fraction=np.clip((u-LEVELS[lower])/(LEVELS[lower+1]-LEVELS[lower]),0,1)
    return low+fraction*(high-low)


def pit(values,target,rng):
    index=np.clip((values<=target[None]).sum(0)-1,0,len(LEVELS)-2)
    low=np.take_along_axis(values,index[None],axis=0)[0]
    high=np.take_along_axis(values,(index+1)[None],axis=0)[0]
    fraction=np.clip((target-low)/np.maximum(high-low,1e-12),0,1)
    u=LEVELS[index]+fraction*(LEVELS[index+1]-LEVELS[index])
    u=np.where(target<values[0],0,np.where(target>values[-1],1,u))
    equal=np.isclose(values,target[None],rtol=0,atol=1e-10);ties=equal.any(0)
    left=np.where(equal,LEVELS[:,None,None],np.inf).min(0)
    right=np.where(equal,LEVELS[:,None,None],-np.inf).max(0)
    left=np.where(equal[0],0,left);right=np.where(equal[-1],1,right)
    left=np.where(ties,left,0);right=np.where(ties,right,0)
    return np.where(ties,left+rng.random(target.shape)*(right-left),u)


def fit(data,days,out):
    x,delta=build_arrays(data['data'],days);target=x[:,:5]+delta
    states=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in days]);hi=states.max(0)*1.1
    subsets=[]
    for m in range(5):
        selected=np.random.RandomState(m).choice(len(days),int(.8*len(days)),replace=False)
        subsets.append((selected[:,None]*95+np.arange(95)).reshape(-1))
    def job(m,q,no3=None):
        row=[]
        for j in range(5):
            if j==1 and no3 is not None:row.append(no3);continue
            xx=x.copy();constraint=np.zeros(x.shape[1],dtype=int)
            if j==0:constraint[0]=1;constraint[2:5]=-1;constraint[14:24]=-1;xx[:,1]=0
            elif j>=2:constraint[0]=-1;constraint[2:5]=1;constraint[14:24]=1;xx[:,1]=0
            else:constraint[1]=1
            model=HGB(loss='quantile',quantile=q,max_iter=200,learning_rate=.05,max_leaf_nodes=15,
                min_samples_leaf=40,l2_regularization=1,monotonic_cst=constraint,early_stopping=False,random_state=100*m+j)
            model.fit(xx[subsets[m]],target[subsets[m],j])
            if j!=1:
                assert not any(np.any(tree[0].nodes['feature_idx'][tree[0].nodes['is_leaf']==0]==1) for tree in model._predictors)
            row.append(DirectDeltaRegressor(model,j))
        print('TRAINED',m,q,flush=True);return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        median=list(pool.map(lambda m:job(m,.5),range(5)))
        rows={.5:median}
        for q in [.05,.25,.75,.95]:
            rows[q]=list(pool.map(lambda m:job(m,q,median[m][1]),range(5)))
            # Keep a completed quantile checkpoint even if a later fit fails.
            with (out/f'quantile_{q}.pkl').open('wb') as f:pickle.dump({'models':rows[q],'hi':hi,'bias':np.zeros(5)},f)
    with (out/'quantile_0.5.pkl').open('wb') as f:pickle.dump({'models':median,'hi':hi,'bias':np.zeros(5)},f)
    return x,hi


class QuantileDynamics:
    def __init__(self,out):
        self.models=[GBMDynamics(out/f'quantile_{q}.pkl') for q in LEVELS]
        self.hi=self.models[0].hi;self.n_members=5

    def quantiles(self,x,members):
        values=np.stack([x[:,:5]+model.delta(x,members) for model in self.models])
        return np.clip(np.sort(values,axis=0),0,self.hi)


def paths(bank,days,block,draws=8,uniform=False):
    values=[]
    for m in range(5):
        for draw in range(draws):
            for day in days:
                seed=int.from_bytes(hashlib.sha256(f'e38:{day}:{m}:{draw}'.encode()).digest()[:4],'little')
                rng=np.random.default_rng(seed)
                if uniform:u=rng.random((95,5))
                else:
                    count=int(np.ceil(95/block))
                    start=rng.integers(0,len(bank[m])//95,size=count)*95+rng.integers(0,96-block,size=count)
                    assert np.all(start//95==(start+block-1)//95)
                    index=(start[:,None]+np.arange(block)).reshape(-1)[:95];u=bank[m,index].copy()
                u[:,1]=.5;values.append(u)
    return np.stack(values)


def rollout(dyn,batch,u,alpha,draws=8,air_scale=1.):
    true,exo,zones=batch;n=len(true)
    members=np.repeat(np.arange(5),draws*n);state=np.tile(true[:,0],(5*draws,1))
    exo=np.tile(exo,(5*draws,1,1));zones=zones*air_scale
    zones*=np.minimum(1,9000/np.maximum(zones.sum(-1,keepdims=True),1))
    zones=np.tile(zones,(5*draws,1,1));values=[]
    for t in range(95):
        x=features(state,exo[:,t],zones[:,t],zones[:,max(0,t-1)])
        sampled=interpolate(dyn.quantiles(x,members),u[:,t])
        state=np.clip(alpha*state+(1-alpha)*sampled,0,dyn.hi);values.append(state.copy())
    return np.stack(values,1).reshape(5,draws,n,95,5)


def checks(dyn,x,strict_order=False):
    rng=np.random.default_rng(38);values=np.sort(rng.uniform(0,5,(5,80,5)),axis=0);u=rng.random((80,5))
    expected=np.array([[np.interp(u[i,j],LEVELS,values[:,i,j]) for j in range(5)] for i in range(80)])
    np.testing.assert_allclose(interpolate(values,u),expected,atol=1e-12)
    atom=np.array([0,0,1,2,2.])[:,None,None]*np.ones((5,80,5))
    atom_u=pit(atom,np.zeros((80,5)),rng);assert atom_u.min()>=0 and atom_u.max()<=.25
    atom_u=pit(atom,np.ones((80,5))*2,rng);assert atom_u.min()>=.75 and atom_u.max()<=1
    probe=x[np.linspace(0,len(x)-1,256,dtype=int)].copy();probe[:128,:5]*=1.05
    members=np.arange(len(probe))%5;maximum=0.
    for model in dyn.models:
        slow=np.zeros((len(probe),5))
        for m in range(5):
            mask=members==m
            for j in range(5):slow[mask,j]=model.models[m][j].predict(probe[mask])
        fast=model.delta(probe,members);np.testing.assert_allclose(fast,slow,rtol=1e-9,atol=1e-9)
        maximum=max(maximum,float(np.abs(fast-slow).max()))
    small=probe[:64];ids=members[:64];base=dyn.quantiles(small,ids);order=0.
    for feature in [0,2,3,4,*range(14,24)]:
        changed=small.copy();changed[:,feature]+=.2*np.maximum(np.abs(changed[:,feature]),.1)
        higher=dyn.quantiles(changed,ids)
        if feature==0:
            violation=max(float(np.maximum(base[...,0]-higher[...,0],0).max()),float(np.maximum(higher[...,2:]-base[...,2:],0).max()))
        else:
            violation=max(float(np.maximum(higher[...,0]-base[...,0],0).max()),float(np.maximum(base[...,2:]-higher[...,2:],0).max()))
        order=max(order,violation)
    if strict_order:assert order<1e-8
    return {'interpolation_numpy_reference':True,'endpoint_atom_cdf_intervals':True,
        'fast_sklearn_max_difference':maximum,'coordinate_order_max_violation':order}


def main(training_context=None,strict_order=False):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);os.environ['DARROW_FAST_DYNAMICS']='1'
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    raw=pd.read_pickle(a.data);ref=json.loads(a.reference_summary.read_text());fit_days=ref['fit_days'];cal=ref['calibration_days']
    with (nullcontext() if training_context is None else training_context()) as training_info:
        x,hi=fit(raw,fit_days,a.out)
    dyn=QuantileDynamics(a.out);verification=checks(dyn,x,strict_order=strict_order)
    if training_info is not None:verification['leaf_bound_enforcement']=training_info
    xc,delta=build_arrays(raw['data'],cal);members=np.repeat(np.arange(5),len(xc))
    values=dyn.quantiles(np.tile(xc,(5,1)),members);target=np.tile(xc[:,:5]+delta,(5,1))
    bank=pit(values,target,np.random.default_rng(3801)).reshape(5,len(xc),5);bank[...,1]=.5
    np.savez_compressed(a.out/'pit_bank.npz',pit=bank)
    verification['inner_central90_coverage']=((target>=values[0])&(target<=values[-1])).mean(0).tolist()
    verification['inner_outside_quantile_range']=((target<values[0])|(target>values[-1])).mean(0).tolist()
    np.testing.assert_array_equal(paths(bank,cal,4),paths(bank,cal,4));verification['noise_reproducibility']=True
    sd=np.concatenate([raw['data'].loc[d][STATE].to_numpy(float) for d in fit_days]).std(0)
    batch=arrays(raw['data'],cal);inner={};settings={}
    for mode,block in [('uniform',1),('pit',1),('pit',4),('pit',16),('pit',95)]:
        noise=paths(bank,cal,block,uniform=mode=='uniform')
        for alpha in [0,.25,.5,.75]:
            key=f'{mode}_b{block}_a{alpha}';settings[key]=(mode,block,alpha)
            inner[key]=summarize(rollout(dyn,batch,noise,alpha),batch[0],sd)
            print('INNER',key,inner[key]['proper_selection_score'],flush=True)
    selected=min(inner,key=lambda key:inner[key]['proper_selection_score']);days=raw['days']['val'];batch=arrays(raw['data'],days)
    validation={}
    for key in dict.fromkeys(['uniform_b1_a0',selected]):
        mode,block,alpha=settings[key];noise=paths(bank,days,block,uniform=mode=='uniform')
        traj=rollout(dyn,batch,noise,alpha);score=summarize(traj,batch[0],sd)
        score['event_diagnostics']=event_diagnostics(traj,batch[0])
        low=rollout(dyn,batch,noise,alpha,air_scale=.75);high=rollout(dyn,batch,noise,alpha,air_scale=1.25)
        score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
            'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())};validation[key]=score
    mode,block,alpha=settings[selected]
    sensitivity=summarize(rollout(dyn,batch,paths(bank,days,block,draws=16,uniform=mode=='uniform'),alpha,draws=16),batch[0],sd)
    result={'selected':selected,'gate':gate(validation[selected],ref['validation']['old']),'inner':inner,
        'validation':validation,'sixteen_draw_sensitivity':sensitivity,'checks':verification,'test_evaluated':False,
        'fit_days':fit_days,'calibration_days':cal,'code_commit':commit,'quantile_levels':LEVELS.tolist(),
        'model_hashes':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in a.out.glob('*.pkl')}}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('RESULT',selected,json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
