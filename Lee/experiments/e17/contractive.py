"""Fixed contractive state feedback plus monotone exogenous forcing."""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
from cooperative import arrays,recursive,evaluate,calibrate,gate
from surrogate_dw import GBMDynamics,STATE,build_arrays
from direct_dynamics import LeakyDirectDeltaRegressor


def train(data,days,path,coefficients):
    x,delta=build_arrays(data['data'],days)
    target=x[:,:5]+delta-coefficients*x[:,:5]
    xx=x.copy();xx[:,:5]=0
    models=[]
    for m in range(5):
        idx=np.random.RandomState(m).choice(len(x),int(.8*len(x)),replace=False)
        row=[]
        for j in range(5):
            c=np.zeros(x.shape[1],dtype=int)
            if j==0:c[14:24]=-1
            if j>=2:c[14:24]=1
            model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,
                min_samples_leaf=40,l2_regularization=1,monotonic_cst=c,early_stopping=False,random_state=m+j)
            model.fit(xx[idx],target[idx,j])
            assert not any(np.any(tree[0].nodes['feature_idx'][tree[0].nodes['is_leaf']==0]<5) for tree in model._predictors)
            row.append(LeakyDirectDeltaRegressor(model,j,float(coefficients[j])))
        models.append(row)
    fit=np.concatenate([data['data'].loc[day][STATE].to_numpy(float) for day in days])
    with path.open('wb') as f:pickle.dump({'models':models,'hi':fit.max(0)*1.1,'bias':np.zeros(5)},f)


def verify(path,data,days,coefficients):
    os.environ['DARROW_FAST_DYNAMICS']='0';slow=GBMDynamics(path)
    os.environ['DARROW_FAST_DYNAMICS']='1';fast=GBMDynamics(path)
    x,_=build_arrays(data['data'],days)
    error=0.
    for m in range(5):
        expected=slow.delta(x,np.full(len(x),m));actual=fast.delta(x,np.full(len(x),m))
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)
        error=max(error,float(np.max(np.abs(actual-expected))))
    initial,exo,zone=arrays(data['data'],days[:4])
    a=initial[:,0].copy();b=a+.01
    bound=np.full_like(a,.01)
    for t in range(95):
        a=fast.step(a,exo[:,t],zone[:,t],zone[:,max(0,t-1)],np.zeros(len(a),dtype=int))
        b=fast.step(b,exo[:,t],zone[:,t],zone[:,max(0,t-1)],np.zeros(len(a),dtype=int))
        bound*=coefficients
        assert np.all(np.abs(a-b)<=bound+1e-10)
    return {'max_fast_prediction_difference':error,'contraction_95_steps_verified':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.environ['DARROW_FAST_DYNAMICS']='1'
    data=pd.read_pickle(a.data)
    fit,cal=data['days']['train'][:-30],data['days']['train'][-30:]
    sd=np.concatenate([data['data'].loc[day][STATE].to_numpy(float) for day in fit]).std(0)
    paths={};calibration={};verification={};inner={}
    for nh4,do in [(.95,.2),(.95,.8),(.99,.2),(.99,.8)]:
        name=f'n{round(nh4*100)}_d{round(do*100)}'
        coefficients=np.array([nh4,.99,do,do,do])
        path=a.out/f'{name}.pkl'
        train(data,fit,path,coefficients)
        verification[name]=verify(path,data,cal,coefficients)
        paths[name]=path
        cp=a.out/f'{name}_cal.pkl'
        calibration[name]=calibrate(path,data,cal,sd,cp)
        paths[name+'_cal']=cp
        (a.out/f'{name}_calibration.json').write_text(json.dumps(calibration[name],indent=2))
        print('FINISHED',name,calibration[name]['initial_score'],calibration[name]['best_score'],flush=True)
    for name,path in paths.items():inner[name]=evaluate(GBMDynamics(path),data,cal,sd)
    def selection(name):
        v=inner[name]
        return v['normalized_mae']+.5*(v['nh4_violation']['absolute_error_pp']+v['do_violation']['absolute_error_pp'])/100
    selected=min(inner,key=selection)
    scores={'old':evaluate(GBMDynamics(a.old),data,data['days']['val'],sd)}
    for name,path in paths.items():
        scores[name]=evaluate(GBMDynamics(path),data,data['days']['val'],sd)
        v=scores[name]
        print('VALIDATION',name,v['normalized_mae'],v['mae']['nh4'],v['nh4_violation']['pred'],v['do_violation']['pred'],flush=True)
    result={'selected_by_inner_calibration':selected,'gate':gate(scores[selected],scores['old']),
        'inner':inner,'validation':scores,'calibration':calibration,'verification':verification,
        'model_paths':{k:str(v.resolve()) for k,v in paths.items()},'model_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        'test_evaluated':False,'fit_days':fit,'calibration_days':cal,'validation_days':data['days']['val'],
        'code_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',selected,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
