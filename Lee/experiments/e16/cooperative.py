"""Cooperative concentration models and train-only recursive bias calibration."""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
from direct_dynamics import DirectDeltaRegressor,PersistenceRegressor
from surrogate_dw import GBMDynamics,STATE,EXO,ZONES,build_arrays


def arrays(table,days):
    groups=[table.loc[day] for day in days]
    return (np.stack([g[STATE].to_numpy(float) for g in groups]),
            np.stack([g[EXO].to_numpy(float) for g in groups]),
            np.stack([g[ZONES].to_numpy(float) for g in groups]))


def recursive(dyn,batch,scale=1.,bias=None):
    true,exo,zone=batch
    n=len(true)
    members=np.repeat(np.arange(dyn.n_members),n)
    state=np.tile(true[:,0],(dyn.n_members,1))
    exo=np.tile(exo,(dyn.n_members,1,1))
    zone=np.tile(np.clip(zone*scale,0,None),(dyn.n_members,1,1))
    values=[]
    for t in range(95):
        state=dyn.step(state,exo[:,t],zone[:,t],zone[:,max(0,t-1)],members)
        if bias is not None:
            state=np.clip(state+bias,0,dyn.hi)
        values.append(state.copy())
    return np.stack(values,1).reshape(dyn.n_members,n,95,5)


def metric(traj,true,sd):
    error=np.abs(traj-true[None,:,1:])
    mae=error.mean((0,1,2))
    pn=(traj[...,0]>4).mean((1,2))*100
    pd_=(traj[...,2:].max(-1)>2).mean((1,2))*100
    tn=float((true[:,1:,0]>4).mean()*100)
    td=float((true[:,1:,2:].max(-1)>2).mean()*100)
    return {'normalized_mae':float((mae/sd).mean()),'mae':dict(zip(STATE,mae.tolist())),
        'nh4_violation':{'pred':float(pn.mean()),'actual':tn,'member_pred':pn.tolist(),'absolute_error_pp':float(abs(pn.mean()-tn))},
        'do_violation':{'pred':float(pd_.mean()),'actual':td,'member_pred':pd_.tolist(),'absolute_error_pp':float(abs(pd_.mean()-td))},
        'horizon_mae':{str(k):dict(zip(STATE,error[:,:,:k].mean((0,1,2)).tolist())) for k in [1,4,16,48,95]}}


def evaluate(dyn,data,days,sd):
    batch=arrays(data['data'],days)
    traj=recursive(dyn,batch)
    result=metric(traj,batch[0],sd)
    x,delta=build_arrays(data['data'],days)
    pred=np.mean([dyn.delta(x,np.full(len(x),m)) for m in range(dyn.n_members)],0)
    true=x[:,:5]+delta
    result['one_step_mae']=dict(zip(STATE,np.abs(pred-delta).mean(0).tolist()))
    result['next_concentration_r2']=dict(zip(STATE,(1-((pred-delta)**2).sum(0)/((true-true.mean(0))**2).sum(0)).tolist()))
    # The same scalar applies to all zone inputs. No action clipping is used in
    # this mathematical order check; the exercised air range is reported.
    low=recursive(dyn,batch,.75)
    high=recursive(dyn,batch,1.25)
    result['recursive_order']={
        'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
        'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max()),
        'low_nh4_mean':float(low[...,0].mean()),'high_nh4_mean':float(high[...,0].mean()),
        'low_do_means':low[...,2:].mean((0,1,2)).tolist(),'high_do_means':high[...,2:].mean((0,1,2)).tolist()}
    return result


def train(data,days,path,leaves):
    x,delta=build_arrays(data['data'],days)
    target=x[:,:5]+delta
    models=[]
    for member in range(5):
        idx=np.random.RandomState(member).choice(len(x),int(.8*len(x)),replace=False)
        row=[]
        for j in range(5):
            xx=x.copy()
            c=np.zeros(x.shape[1],dtype=int)
            if j==0:
                c[0]=1;c[2:5]=-1;c[14:24]=-1;xx[:,1]=0
            elif j>=2:
                c[0]=-1;c[2:5]=1;c[14:24]=1;xx[:,1]=0
            else:
                c[1]=1
            model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=leaves,
                min_samples_leaf=40,l2_regularization=1,monotonic_cst=c,early_stopping=False,random_state=member+j)
            model.fit(xx[idx],target[idx,j])
            if j!=1:
                assert not any(np.any(tree[0].nodes['feature_idx'][tree[0].nodes['is_leaf']==0]==1) for tree in model._predictors)
            row.append(DirectDeltaRegressor(model,j))
        models.append(row)
        print('TRAINED',leaves,member,flush=True)
    states=np.concatenate([data['data'].loc[day][STATE].to_numpy(float) for day in days])
    with path.open('wb') as f:pickle.dump({'models':models,'hi':states.max(0)*1.1,'bias':np.zeros(5)},f)


def calibrate(path,data,days,sd,out):
    dyn=GBMDynamics(path)
    batch=arrays(data['data'],days)
    x,delta=build_arrays(data['data'],days)
    prediction=np.mean([dyn.delta(x,np.full(len(x),m)) for m in range(5)],0)
    bound=np.clip(np.quantile(np.abs(prediction-delta),.9,axis=0),[.02,.02,.01,.01,.01],[1,1,.3,.3,.3])
    target_n=(batch[0][:,1:,0]>4).mean()
    target_d=(batch[0][:,1:,2:].max(-1)>2).mean()
    trace=[]
    best=[float('inf'),np.zeros(5)]
    def objective(u):
        traj=recursive(dyn,batch,bias=u*bound)
        mae=np.abs(traj-batch[0][None,:,1:]).mean((0,1,2))
        sn=expit((traj[...,0]-4)/.25).mean()
        ds=expit((traj[...,2:].max(-1)-2)/.1).mean()
        score=float((mae/sd).mean()+.5*(abs(sn-target_n)+abs(ds-target_d)))
        trace.append({'u':u.tolist(),'score':score,'soft_nh4_frequency':float(sn),'soft_do_frequency':float(ds)})
        if score<best[0]:best[:]=[score,u.copy()]
        return score
    start=time.monotonic()
    initial=objective(np.zeros(5))
    optimization=minimize(objective,np.zeros(5),method='Powell',bounds=[(-1,1)]*5,
                          options={'maxfev':120,'maxiter':8,'xtol':.04,'ftol':.002})
    with path.open('rb') as f:payload=pickle.load(f)
    payload['bias']=best[1]*bound
    with out.open('wb') as f:pickle.dump(payload,f)
    return {'initial_score':initial,'best_score':best[0],'bias':payload['bias'].tolist(),'bounds':bound.tolist(),
        'optimizer_success':bool(optimization.success),'optimizer_message':str(optimization.message),
        'seconds':time.monotonic()-start,'trace':trace}


def gate(candidate,reference):
    n=candidate['nh4_violation'];d=candidate['do_violation']
    members=sum(abs(a-n['actual'])<=5 and abs(b-d['actual'])<=1 for a,b in zip(n['member_pred'],d['member_pred']))
    checks={'normalized_error_improves_10pct':candidate['normalized_mae']<=.9*reference['normalized_mae'],
        'nh4_error_improves_10pct':candidate['mae']['nh4']<=.9*reference['mae']['nh4'],
        'nh4_calibration':n['absolute_error_pp']<=5 and n['absolute_error_pp']<=reference['nh4_violation']['absolute_error_pp'],
        'do_calibration':d['absolute_error_pp']<=1 and d['absolute_error_pp']<=reference['do_violation']['absolute_error_pp'],
        'four_members_calibrated':members>=4,
        'recursive_action_order':candidate['recursive_order']['nh4_max_order_violation']<1e-8 and candidate['recursive_order']['do_max_order_violation']<1e-8}
    return {'passed':all(checks.values()),'checks':checks,'members_calibrated':members}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--direct',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ['DARROW_FAST_DYNAMICS']='1'
    data=pd.read_pickle(a.data)
    fit_days,cal_days=data['days']['train'][:-30],data['days']['train'][-30:]
    sd=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in fit_days]).std(0)
    paths={'old':a.old,'direct7':a.direct}
    calibration={}
    for leaves in [7,15]:
        path=a.out/f'cooperative_{leaves}.pkl'
        train(data,fit_days,path,leaves)
        paths[f'cooperative_{leaves}']=path
        calibrated=a.out/f'calibrated_{leaves}.pkl'
        calibration[str(leaves)]=calibrate(path,data,cal_days,sd,calibrated)
        paths[f'calibrated_{leaves}']=calibrated
        (a.out/f'calibration_{leaves}.json').write_text(json.dumps(calibration[str(leaves)],indent=2))
        print('CALIBRATED',leaves,calibration[str(leaves)]['initial_score'],calibration[str(leaves)]['best_score'],flush=True)
    inner={}
    for name,path in paths.items():
        inner[name]=evaluate(GBMDynamics(path),data,cal_days,sd)
    eligible=[k for k in paths if k not in ['old','direct7']]
    def selection_score(name):
        v=inner[name]
        return v['normalized_mae']+.5*(v['nh4_violation']['absolute_error_pp']+v['do_violation']['absolute_error_pp'])/100
    selected=min(eligible,key=selection_score)
    scores={}
    for name,path in paths.items():
        scores[name]=evaluate(GBMDynamics(path),data,data['days']['val'],sd)
        print('VALIDATION',name,json.dumps(scores[name]),flush=True)
    # Persistence remains a diagnostic with no usable control response.
    with (a.out/'persistence.pkl').open('wb') as f:
        pickle.dump({'models':[[PersistenceRegressor() for _ in STATE]],'hi':np.full(5,np.inf)},f)
    os.environ['DARROW_FAST_DYNAMICS']='0'
    scores['persistence']=evaluate(GBMDynamics(a.out/'persistence.pkl'),data,data['days']['val'],sd)
    result={'selected_by_inner_calibration':selected,'gate':gate(scores[selected],scores['old']),
        'fit_days':fit_days,'calibration_days':cal_days,'validation_days':data['days']['val'],
        'test_evaluated':False,'inner':inner,'validation':scores,'model_paths':{k:str(v.resolve()) for k,v in paths.items()},
        'model_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        'calibration':calibration,'code_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',selected,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
