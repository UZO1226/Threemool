"""Separate forecast drift from one-step error and inspect action support."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e18'))
from surrogate_dw import GBMDynamics,STATE,build_arrays,features
from cooperative import arrays
from innovations import noise_paths,summarize


def reset_rollout(dyn,batch,noise,scale,horizon,draws=8):
    true,exo,zone=batch;n=len(true)
    members=np.repeat(np.arange(dyn.n_members),draws*n)
    state=np.tile(true[:,0],(dyn.n_members*draws,1))
    exo=np.tile(exo,(dyn.n_members*draws,1,1));zone=np.tile(zone,(dyn.n_members*draws,1,1))
    result=[]
    for t in range(95):
        if t%horizon==0:state=np.tile(true[:,t],(dyn.n_members*draws,1))
        x=features(state,exo[:,t],zone[:,t],zone[:,max(0,t-1)])
        state=np.clip(state+dyn.delta(x,members)+noise[:,t]*scale,0,dyn.hi)
        result.append(state.copy())
    return np.stack(result,1).reshape(dyn.n_members,draws,n,95,5)


def support(data,fit,cal):
    def xy(days):
        x,_=build_arrays(data['data'],days)
        return np.column_stack([x[:,:14],x[:,19:24]]),x[:,14:19].sum(1)*1000
    xf,yf=xy(fit);xc,yc=xy(cal);xv,yv=xy(data['days']['val'])
    models=[HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=200,max_leaf_nodes=15,
        learning_rate=.05,min_samples_leaf=40,l2_regularization=1,early_stopping=False,random_state=2026).fit(xf,yf)
        for q in [.05,.95]]
    low,high=[m.predict(xc) for m in models];lo=np.minimum(low,high);hi=np.maximum(low,high)
    error=np.maximum(lo-yc,yc-hi)
    correction=max(0,float(np.quantile(error,.9,method='higher')))
    low,high=[m.predict(xv) for m in models];lo=np.minimum(low,high)-correction;hi=np.maximum(low,high)+correction
    result={'calibration_adjustment_Nm3_h':correction,'development_empirical_coverage_pct':float(((yv>=lo)&(yv<=hi)).mean()*100),
        'development_median_interval_width_Nm3_h':float(np.median(hi-lo)),
        'counterfactual_outside_interval_pct':{str(s):float(((np.clip(yv*s,0,9000)<lo)|(np.clip(yv*s,0,9000)>hi)).mean()*100) for s in [.75,1.25]},
        'caveat':'Logged-state descriptive intervals, not identified causal effects or guaranteed conditional coverage'}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--innovation-summary',type=Path,required=True);p.add_argument('--residual-bank',type=Path,required=True)
    p.add_argument('--contractive-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();os.environ['DARROW_FAST_DYNAMICS']='1'
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    inv=json.loads(a.innovation_summary.read_text());ref=json.loads(a.contractive_summary.read_text())
    assert hashlib.sha256(a.residual_bank.read_bytes()).hexdigest()==inv['residual_bank_sha256']
    data=pd.read_pickle(a.data);dyn=GBMDynamics(inv['base_model_path']);bank=np.load(a.residual_bank)['residual']
    batch=arrays(data['data'],data['days']['val']);noise=noise_paths(bank,data['days']['val'])
    sd=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in ref['fit_days']]).std(0)
    scores={}
    for horizon in [1,4,16,48,95]:
        traj=reset_rollout(dyn,batch,noise,np.array(inv['scale_vector']),horizon)
        scores[str(horizon)]=summarize(traj,batch[0],sd)
        v=scores[str(horizon)]
        print('HORIZON',horizon,v['mae']['nh4'],v['nh4_violation']['pred'],v['do_violation']['pred'],flush=True)
    for key in ['normalized_mae','nh4_brier','do_brier']:
        np.testing.assert_allclose(scores['95'][key],inv['validation'][inv['selected_scale']][key],rtol=1e-10,atol=1e-10)
    air=support(data,ref['fit_days'],ref['calibration_days'])
    result={'horizon_scores':scores,'logged_action_support':air,'full_horizon_reproduces_e18':True,
        'test_evaluated':False,'code_commit':commit,
        'interpretation_boundary':'Resetting observed concentrations is a forecast diagnostic, not policy evaluation'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SUPPORT',json.dumps(air),flush=True)


if __name__=='__main__':main()
