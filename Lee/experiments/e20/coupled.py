"""Fit contractive coupled state matrices and score probabilistic innovations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e18'))
from surrogate_dw import GBMDynamics,STATE,build_arrays
from direct_dynamics import CoupledForcingRegressor
from cooperative import arrays,recursive,calibrate,gate
from innovations import bank,noise_paths,rollout,summarize


def fit(data,days,path,rho,sd):
    x,delta=build_arrays(data['data'],days)
    y=x[:,:5]+delta
    sign=np.array([-1,1,1,1,1])
    z=x[:,:5]*sign/sd
    exo=x[:,5:]
    models=[];matrices=[];solver=[]
    for m in range(5):
        idx=np.random.RandomState(m).choice(len(x),int(.8*len(x)),replace=False)
        xm=exo[idx].mean(0);xs=exo[idx].std(0)+1e-9
        design=np.column_stack([z,(exo-xm)/xs,np.ones(len(x))])
        D=design[idx];gram=D.T@D/len(idx)
        row=[];A=[];checks=[]
        for j in range(5):
            target=y[idx,j]*sign[j]/sd[j]
            rhs=D.T@target/len(idx)
            penalty=np.eye(D.shape[1])*.001;penalty[-1,-1]=0
            gram_penalized=gram+penalty
            start=np.zeros(D.shape[1]);start[j]=min(.9,rho);start[-1]=target.mean()-(D@start).mean()
            allowed=[j] if j==1 else [0,2,3,4]
            bounds=[(0,rho) if k in allowed else (0,0) for k in range(5)]
            bounds+=[(0,None) if j!=1 and k>=9 else (None,None) for k in range(19)]
            bounds.append((None,None))
            def fun(v):return float(.5*v@gram_penalized@v-rhs@v)
            def jac(v):return gram_penalized@v-rhs
            constraint={'type':'ineq','fun':lambda v:rho-v[:5].sum(),
                        'jac':lambda v:np.r_[-np.ones(5),np.zeros(len(v)-5)]}
            result=minimize(fun,start,jac=jac,bounds=bounds,constraints=[constraint],method='SLSQP',
                options={'maxiter':200,'ftol':1e-9})
            aa=np.maximum(result.x[:5],0)
            assert aa.sum()<=rho+1e-7
            if aa.sum()>rho:aa*=rho/aa.sum()
            A.append(aa)
            raw=sd[j]*sign[j]*aa*sign/sd
            target_raw=y[:,j]-x[:,:5]@raw
            xx=x.copy();xx[:,:5]=0
            c=np.zeros(x.shape[1],dtype=int)
            if j==0:c[14:24]=-1
            elif j>=2:c[14:24]=1
            model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,
                min_samples_leaf=40,l2_regularization=1,monotonic_cst=c,early_stopping=False,random_state=m+j)
            model.fit(xx[idx],target_raw[idx])
            assert not any(np.any(tree[0].nodes['feature_idx'][tree[0].nodes['is_leaf']==0]<5) for tree in model._predictors)
            row.append(CoupledForcingRegressor(model,j,raw))
            checks.append({'success':bool(result.success),'iterations':int(result.nit),'message':str(result.message)})
        models.append(row);matrices.append(np.asarray(A).tolist());solver.append(checks)
        print('FITTED',rho,m,flush=True)
    state=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in days])
    with path.open('wb') as f:pickle.dump({'models':models,'hi':state.max(0)*1.1,'bias':np.zeros(5)},f)
    return {'transformed_matrices':matrices,'solver':solver}


def verify(path,data,days,sd,rho):
    os.environ['DARROW_FAST_DYNAMICS']='0';slow=GBMDynamics(path)
    os.environ['DARROW_FAST_DYNAMICS']='1';fast=GBMDynamics(path)
    x,_=build_arrays(data['data'],days)
    for m in range(5):
        np.testing.assert_allclose(slow.delta(x,np.full(len(x),m)),fast.delta(x,np.full(len(x),m)),rtol=1e-10,atol=1e-10)
    s,exo,zone=arrays(data['data'],days[:4])
    for m in range(5):
        a=s[:,0].copy();b=a+.01*sd
        for t in range(95):
            a=fast.step(a,exo[:,t],zone[:,t],zone[:,max(0,t-1)],np.full(len(a),m))
            b=fast.step(b,exo[:,t],zone[:,t],zone[:,max(0,t-1)],np.full(len(a),m))
            assert np.max(np.abs(a-b)/sd)<=.01*rho**(t+1)+1e-9
    return {'fast_matches_sklearn':True,'weighted_contraction_all_members':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    os.environ['DARROW_FAST_DYNAMICS']='1'
    ref=json.loads(a.reference_summary.read_text());data=pd.read_pickle(a.data)
    fit_days,cal=ref['fit_days'],ref['calibration_days']
    sd=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in fit_days]).std(0)
    paths={};metadata={};inner_mean={};batch=arrays(data['data'],cal)
    for rho in [.95,.995]:
        name=str(rho);raw=a.out/f'coupled_{name}.pkl'
        metadata[name]=fit(data,fit_days,raw,rho,sd)
        calibrated=a.out/f'coupled_{name}_cal.pkl'
        metadata[name]['calibration']=calibrate(raw,data,cal,sd,calibrated)
        metadata[name]['verification']=verify(calibrated,data,cal,sd,rho)
        paths[name]=calibrated
        dyn=GBMDynamics(calibrated)
        inner_mean[name]=summarize(recursive(dyn,batch)[:,None],batch[0],sd)
        print('CALIBRATED',rho,inner_mean[name]['proper_selection_score'],flush=True)
    selected=min(inner_mean,key=lambda key:inner_mean[key]['proper_selection_score'])
    dyn=GBMDynamics(paths[selected]);residual=bank(dyn,data,cal)
    np.savez_compressed(a.out/'residual_bank.npz',residual=residual)
    noise=noise_paths(residual,cal);inner={};scale_vectors={}
    for n in [0,.25,.5,1]:
        for d in [0,.5,1,1.5]:
            key=f'n{n}_d{d}';scale=np.array([n,0,d,d,d]);scale_vectors[key]=scale
            inner[key]=summarize(rollout(dyn,batch,noise,scale),batch[0],sd)
            print('INNER',key,inner[key]['proper_selection_score'],flush=True)
    winner=min(inner,key=lambda key:inner[key]['proper_selection_score']);scale=scale_vectors[winner]
    vb=arrays(data['data'],data['days']['val']);vn=noise_paths(residual,data['days']['val'])
    score=summarize(rollout(dyn,vb,vn,scale),vb[0],sd)
    low=rollout(dyn,vb,vn,scale,air_scale=.75);high=rollout(dyn,vb,vn,scale,air_scale=1.25)
    score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
        'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
    deterministic=summarize(recursive(dyn,vb)[:,None],vb[0],sd)
    sensitivity=summarize(rollout(dyn,vb,noise_paths(residual,data['days']['val'],draws=16),scale,draws=16),vb[0],sd)
    result={'selected_mean':selected,'selected_scale':winner,'scale':scale.tolist(),'gate':gate(score,ref['validation']['old']),
        'validation':score,'deterministic_validation':deterministic,'sixteen_draw_sensitivity':sensitivity,
        'inner_mean':inner_mean,'inner_scales':inner,'metadata':metadata,'test_evaluated':False,'code_commit':commit,
        'model_paths':{k:str(v.resolve()) for k,v in paths.items()},'model_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',selected,winner,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
