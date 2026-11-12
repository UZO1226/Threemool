"""Fit-only temperature trend with inner-only joint innovation selection."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e18'))
from surrogate_dw import GBMDynamics,STATE,build_arrays
from direct_dynamics import TemperatureForcingRegressor
from cooperative import arrays,recursive,gate
from innovations import bank,noise_paths,rollout,summarize


def fit_model(source,data,fit,cal,multiplier,out,sd):
    with source.open('rb') as f:payload=pickle.load(f)
    x,delta=build_arrays(data['data'],fit)
    slopes=[]
    for m in range(5):
        alpha=payload['models'][m][0].state_coefficient
        y=x[:,0]+delta[:,0]-alpha*x[:,0]
        idx=np.random.RandomState(m).choice(len(x),int(.8*len(x)),replace=False)
        design=x[:,5:];center=design[idx].mean(0);scale=design[idx].std(0)+1e-9
        ridge=Ridge(alpha=10).fit((design[idx]-center)/scale,y[idx])
        slope=float(ridge.coef_[1]/scale[1])
        slope=float(np.clip(slope,-(1-alpha)*x[idx,0].std()/max(x[idx,6].std(),1e-6),0))*multiplier
        tref=float(x[idx,6].mean())
        xx=x.copy();xx[:,:5]=0;xx[:,6]=0
        constraints=np.zeros(x.shape[1],dtype=int);constraints[14:24]=-1
        tree=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,
            min_samples_leaf=40,l2_regularization=1,monotonic_cst=constraints,early_stopping=False,random_state=m)
        tree.fit(xx[idx],y[idx]-slope*(x[idx,6]-tref))
        assert not any(np.any(tree_[0].nodes['feature_idx'][tree_[0].nodes['is_leaf']==0]==6) for tree_ in tree._predictors)
        payload['models'][m][0]=TemperatureForcingRegressor(tree,0,alpha,slope,tref)
        slopes.append(slope)
    payload['bias'][0]=0
    with out.open('wb') as f:pickle.dump(payload,f)
    dyn=GBMDynamics(out);batch=arrays(data['data'],cal)
    xc,dc=build_arrays(data['data'],cal)
    residual=dc[:,0]-np.mean([dyn.delta(xc,np.full(len(xc),m))[:,0] for m in range(5)],0)
    bound=float(np.clip(np.quantile(np.abs(residual),.9),.02,1))
    target=(batch[0][:,1:,0]>4).mean()
    trace=[]
    def objective(b):
        bias=np.array([b,0,0,0,0])
        traj=recursive(dyn,batch,bias=bias)
        score=float(np.abs(traj[...,0]-batch[0][None,:,1:,0]).mean()/sd[0]/5
            +.5*abs(expit((traj[...,0]-4)/.25).mean()-target))
        trace.append({'bias':float(b),'score':score})
        return score
    objective(0)
    result=minimize_scalar(objective,bounds=(-bound,bound),method='bounded',options={'maxiter':35,'xatol':.002})
    best=min(trace,key=lambda row:row['score'])
    payload['bias'][0]=best['bias']
    with out.open('wb') as f:pickle.dump(payload,f)
    # Verify all five calibrated numeric predictors, including extrapolated T.
    os.environ['DARROW_FAST_DYNAMICS']='0';slow=GBMDynamics(out)
    os.environ['DARROW_FAST_DYNAMICS']='1';fast=GBMDynamics(out)
    probe=xc.copy();probe[:,6]-=5
    for m in range(5):
        np.testing.assert_allclose(fast.delta(probe,np.full(len(probe),m)),slow.delta(probe,np.full(len(probe),m)),rtol=1e-10,atol=1e-10)
    return {'slopes':slopes,'nh4_bias':best['bias'],'calibration_trace':trace,
        'optimizer_success':bool(result.success),'fast_prediction_verified':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--contractive-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.environ['DARROW_FAST_DYNAMICS']='1'
    source=json.loads(a.contractive_summary.read_text());data=pd.read_pickle(a.data)
    old_name=source['selected_by_inner_calibration'];old_path=Path(source['model_paths'][old_name])
    fit,cal=source['fit_days'],source['calibration_days']
    sd=np.concatenate([data['data'].loc[day][STATE].to_numpy(float) for day in fit]).std(0)
    paths={'original':old_path};metadata={}
    for factor in [0,.5,1,2]:
        name=f'temperature_{factor}';path=a.out/f'{name}.pkl'
        metadata[name]=fit_model(old_path,data,fit,cal,factor,path,sd)
        paths[name]=path
        print('FITTED',name,metadata[name]['slopes'],flush=True)
    cal_batch=arrays(data['data'],cal)
    # DO/NO3 models and biases are identical across mean options. Because
    # feedback is diagonal, component reuse is exactly a joint simulation.
    base=GBMDynamics(old_path);base_bank=bank(base,data,cal);common=noise_paths(base_bank,cal)
    do_profiles={scale:rollout(base,cal_batch,common,np.array([0,0,scale,scale,scale])) for scale in [0,.5,1,1.5]}
    inner={};banks={};models={};nh4_profiles={}
    for name,path in paths.items():
        dyn=GBMDynamics(path);models[name]=dyn;b=bank(dyn,data,cal);banks[name]=b
        np.savez_compressed(a.out/f'{name}_residual_bank.npz',residual=b)
        noise=noise_paths(b,cal)
        for scale in [0,.5,1,1.5,2]:
            traj=rollout(dyn,cal_batch,noise,np.array([scale,0,0,0,0]))
            nh4_profiles[(name,scale)]=traj[...,0].copy()
            for ds,dt in do_profiles.items():
                assembled=dt.copy();assembled[...,0]=traj[...,0]
                key=f'{name}:n{scale}:d{ds}'
                v=summarize(assembled,cal_batch[0],sd)
                inner[key]={'mean_option':name,'nh4_scale':scale,'do_scale':ds,
                    'score':v['proper_selection_score'],'normalized_crps':v['normalized_crps'],
                    'nh4_brier':v['nh4_brier'],'do_brier':v['do_brier']}
        print('INNER_COMPLETE',name,flush=True)
    selected=min(inner,key=lambda key:inner[key]['score']);choice=inner[selected]
    name=choice['mean_option'];dyn=models[name]
    scale=np.array([choice['nh4_scale'],0,choice['do_scale'],choice['do_scale'],choice['do_scale']])
    full=rollout(dyn,cal_batch,noise_paths(banks[name],cal),scale)
    expected=do_profiles[choice['do_scale']].copy();expected[...,0]=nh4_profiles[(name,choice['nh4_scale'])]
    np.testing.assert_allclose(full,expected,rtol=1e-10,atol=1e-10)
    batch=arrays(data['data'],data['days']['val']);noise=noise_paths(banks[name],data['days']['val'])
    traj=rollout(dyn,batch,noise,scale);score=summarize(traj,batch[0],sd)
    low=rollout(dyn,batch,noise,scale,air_scale=.75);high=rollout(dyn,batch,noise,scale,air_scale=1.25)
    score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
        'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
    sensitivity=summarize(rollout(dyn,batch,noise_paths(banks[name],data['days']['val'],draws=16),scale,draws=16),batch[0],sd)
    result={'selected':selected,'choice':choice,'gate':gate(score,source['validation']['old']),
        'validation':score,'sixteen_draw_sensitivity':sensitivity,'inner':inner,'mean_models':metadata,
        'model_paths':{k:str(v.resolve()) for k,v in paths.items()},'model_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        'verification':{'separable_profiles_match_joint_rollout':True},'test_evaluated':False,
        'code_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',selected,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
