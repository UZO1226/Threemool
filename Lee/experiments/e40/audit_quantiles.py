"""Reconstruct stochastic quantile forecasts and audit date uncertainty."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e38'))
from quantile_dynamics import QuantileDynamics,LEVELS,paths,rollout,arrays,features,summarize,event_diagnostics
from cooperative import recursive,GBMDynamics

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
os.environ['DARROW_FAST_DYNAMICS']='1'
s=json.loads((a.runs/'summary.json').read_text());ref=json.loads(a.reference_summary.read_text())
for name,digest in s['model_hashes'].items():assert hashlib.sha256((a.runs/name).read_bytes()).hexdigest()==digest
raw=pd.read_pickle(a.data);days=raw['days']['val'];assert s['fit_days']==ref['fit_days']
assert s['calibration_days']==ref['calibration_days'];assert not s['test_evaluated']
sd=np.concatenate([raw['data'].loc[d][['nh4','no3','do1','do2','do3']].to_numpy(float) for d in s['fit_days']]).std(0)
mode,block_text,alpha_text=s['selected'].split('_');block=int(block_text[1:]);alpha=float(alpha_text[1:])
bank=np.load(a.runs/'pit_bank.npz')['pit'];batch=arrays(raw['data'],days)
noise=paths(bank,days,block,uniform=mode=='uniform');dyn=QuantileDynamics(a.runs)
prediction=rollout(dyn,batch,noise,alpha);score=summarize(prediction,batch[0],sd)
np.testing.assert_allclose(score['normalized_mae'],s['validation'][s['selected']]['normalized_mae'],rtol=1e-12)

# Independently use sklearn tree inference and np.interp for a complete path.
truth,exo,zones=batch;state=truth[0,0].copy();slow=[]
for t in range(95):
    x=features(state[None],exo[:1,t],zones[:1,t],zones[:1,max(0,t-1)])
    quantiles=np.array([[state[j]+q.models[0][j].predict(x)[0] for j in range(5)] for q in dyn.models])
    quantiles=np.clip(np.sort(quantiles,axis=0),0,dyn.hi)
    sampled=np.array([np.interp(noise[0,t,j],LEVELS,quantiles[:,j]) for j in range(5)])
    state=np.clip(alpha*state+(1-alpha)*sampled,0,dyn.hi);slow.append(state.copy())
np.testing.assert_allclose(slow,prediction[0,0,0],rtol=1e-9,atol=1e-9)

target=truth[:,1:];mae=np.abs(prediction-target[None,None]).mean((0,1,3))
daily={'normalized_mae':(mae/sd).mean(1),'nh4_mae':mae[:,0],
    'nh4_bias_pp':(prediction[...,0]>4).mean((0,1,3))*100-(target[...,0]>4).mean(1)*100,
    'do_bias_pp':(prediction[...,2:].max(-1)>2).mean((0,1,3))*100-(target[...,2:].max(-1)>2).mean(1)*100}
rng=np.random.default_rng(20261112);starts=rng.integers(0,len(days),size=(10000,8))
indices=((starts[:,:,None]+np.arange(5))%len(days)).reshape(10000,-1)[:,:len(days)]
intervals={key:{'mean':float(value.mean()),'five_day_block_95_ci':np.quantile(value[indices].mean(1),[.025,.975]).tolist()} for key,value in daily.items()}
original=json.loads((ROOT/'Lee/experiments/e16/summary.json').read_text());references={}
for name in ['old','persistence']:
    os.environ['DARROW_FAST_DYNAMICS']='0' if name=='persistence' else '1'
    path=Path(original['model_paths'][name]) if name=='old' else a.runs.parent/'e16_cooperative/persistence.pkl'
    values=recursive(GBMDynamics(path),batch)
    references[name]=summarize(values[:,None],truth,sd)
result={'selected':s['selected'],'checkpoint_hashes_verified':True,'reconstruction_passed':True,
    'independent_full_day_sklearn_numpy_interp_passed':True,'date_intervals':intervals,
    'event_diagnostics':event_diagnostics(prediction,truth),'reference_scores':references,
    'scoring_sd':sd.tolist(),'days':days,'test_evaluated':False,
    'uncertainty_unit':'dates, circular five-day blocks, exploratory reused validation, not policy seeds'}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(intervals),flush=True)
