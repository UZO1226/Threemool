"""Reconstruct selected forecasts and quantify date-block uncertainty."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
from latent import Data,LatentModel,precompute,simulate,forcing,noise_paths,summarize
from cooperative import arrays,recursive,GBMDynamics

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference-summary',type=Path,required=True)
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--old-diagnosis',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
source=json.loads((a.runs/'summary.json').read_text())
reference=json.loads(a.reference_summary.read_text())
old=json.loads(a.old_diagnosis.read_text())
raw=pd.read_pickle(a.data);data=Data(raw,reference['fit_days']);days=raw['days']['val']
assert days==old['days']
models=[]
for member in range(5):
    model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
    model.load_state_dict(torch.load(a.runs/f"epoch{source['selected_epoch']}_m{member}.pt",weights_only=True))
    model.eval();models.append(model)
cache=precompute(models,data,days,16)
residual=np.load(a.runs/'residual_bank.npz')['residual']
noise=noise_paths(residual,days);bias=np.array(source['bias']);scale=np.array(source['scale'])
traj=simulate(cache,bias,noise,scale);truth=cache['true'][:,1:]
score=summarize(traj,cache['true'],data.sd)
np.testing.assert_allclose(score['normalized_mae'],source['validation']['normalized_mae'],rtol=1e-12)
references={}
original=json.loads((ROOT/'Lee/experiments/e16/summary.json').read_text())
for name in ['old','persistence']:
    os.environ['DARROW_FAST_DYNAMICS']='0' if name=='persistence' else '1'
    path=Path(original['model_paths'][name]) if name=='old' else a.runs.parent/'e16_cooperative/persistence.pkl'
    reference_traj=recursive(GBMDynamics(path),arrays(raw['data'],days))
    references[name]=summarize(reference_traj[:,None],cache['true'],data.sd)

# Independent Torch forward and cached NumPy rollout must agree for a full day.
probe=precompute(models,data,days[:1],16);positions=data.positions[days[0]][:95][None]
exo=torch.tensor(data.exohistory(positions,16),dtype=torch.float32)
context=torch.tensor(data.past(positions[:,0],16),dtype=torch.float32)
state=torch.tensor(data.state[positions[:,0]],dtype=torch.float32)
air=torch.tensor(np.log1p(np.concatenate([probe['zone'],probe['previous_zone']],-1)/1000),dtype=torch.float32)
prediction=simulate(probe,np.zeros(5))
with torch.no_grad():
    for member,model in enumerate(models):
        np.testing.assert_allclose(prediction[member,0],model(state,context,exo,air).numpy(),rtol=2e-4,atol=2e-4)

mae=np.abs(traj-truth[None,None]).mean((0,1,3))
daily={'normalized_mae':(mae/data.sd).mean(1),'nh4_mae':mae[:,0],
    'nh4_bias_pp':(traj[...,0]>4).mean((0,1,3))*100-(truth[...,0]>4).mean(1)*100,
    'do_bias_pp':(traj[...,2:].max(-1)>2).mean((0,1,3))*100-(truth[...,2:].max(-1)>2).mean(1)*100}
daily['normalized_mae_difference_from_old']=daily['normalized_mae']-np.array(old['metrics']['old']['per_day_normalized_mae'])
daily['nh4_mae_difference_from_old']=daily['nh4_mae']-np.array(old['metrics']['old']['per_day_mae'])[:,0]
rng=np.random.default_rng(20261112)
starts=rng.integers(0,len(days),size=(10000,8))
indices=((starts[:,:,None]+np.arange(5))%len(days)).reshape(10000,-1)[:,:len(days)]
intervals={key:{'mean':float(value.mean()),'five_day_block_95_ci':np.quantile(value[indices].mean(1),[.025,.975]).tolist()} for key,value in daily.items()}

# Observed-state resets diagnose forecast drift only, never a control policy.
reset={};m,r,d,_,_=traj.shape;innov=noise.reshape(m,r,d,95,5);f=forcing(cache,1.)
for horizon in [1,4,16,48,95]:
    s=np.repeat(cache['initial'][:,None],r,1);values=[]
    for t in range(95):
        if t%horizon==0:s[...,:5]=cache['true'][None,None,:,t]
        s=np.einsum('mdjk,mrdk->mrdj',cache['V'][:,:,t],s)+f[:,None,:,t]+np.r_[bias,0,0]
        s[...,:5]+=innov[:,:,:,t]*scale;s=np.clip(s,0,cache['hi']);values.append(s[...,:5].copy())
    reset[str(horizon)]=summarize(np.stack(values,3),cache['true'],data.sd)
np.testing.assert_allclose(reset['95']['normalized_mae'],score['normalized_mae'],rtol=1e-8)
result={'selected_epoch':source['selected_epoch'],'selected_scale':source['selected_scale'],
    'torch_numpy_agreement_passed':True,'reconstruction_passed':True,'test_evaluated':False,
    'uncertainty_unit':'dates, circular five-day blocks, exploratory reused validation, not policy seeds',
    'date_intervals':intervals,'reset_diagnostic':reset,'daily':{k:v.tolist() for k,v in daily.items()},
    'reference_scores':references}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(intervals),flush=True)
