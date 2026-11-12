"""Reconstruct selected forecasts and quantify date-block uncertainty."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e26'))
from latent import Data,LatentModel,precompute,simulate,forcing,noise_paths,summarize,apply_reaction
from cooperative import arrays,recursive,GBMDynamics
from correlated import event_diagnostics

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference-summary',type=Path,required=True)
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--parent-runs',type=Path)
p.add_argument('--old-diagnosis',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--model-kind',choices=['base','slow','context','kinetic','monod'],default='base')
a=p.parse_args();torch.set_num_threads(1)
source=json.loads((a.runs/'summary.json').read_text())
weights_dir=a.parent_runs or a.runs
reference=json.loads(a.reference_summary.read_text())
old=json.loads(a.old_diagnosis.read_text())
raw=pd.read_pickle(a.data);data=Data(raw,reference['fit_days']);days=raw['days']['val']
scoring_sd=np.array(reference.get('scoring_sd',data.sd))
model_class=LatentModel;cap=.995
if a.model_kind=='slow':
    sys.path.insert(0,str(ROOT/'Lee/experiments/e27'))
    from slow_memory import SlowMemoryModel
    model_class=SlowMemoryModel;cap=SlowMemoryModel.cap
if a.model_kind=='context':
    sys.path.insert(0,str(ROOT/'Lee/experiments/e32'))
    from persistent_context import PersistentContextModel
    model_class=PersistentContextModel
if a.model_kind=='kinetic':
    sys.path.insert(0,str(ROOT/'Lee/experiments/e33'))
    from kinetic_scaling import KineticScalingModel
    model_class=lambda sd,mean,mean_air,hi:KineticScalingModel(sd,mean,mean_air,hi,data.em[1],data.es[1])
    cap=None
if a.model_kind=='monod':
    sys.path.insert(0,str(ROOT/'Lee/experiments/e39'))
    from monod_hybrid import MonodModel
    model_class=lambda sd,mean,mean_air,hi:MonodModel(sd,mean,mean_air,hi,data.em[1],data.es[1])
    cap=None
assert source.get('state_row_sum_cap',.995)==cap
assert source.get('model_kind','slow' if cap is not None and cap>1 else 'base')==a.model_kind
assert days==old['days']
models=[]
for member in range(5):
    model=model_class(data.sd,data.mean,data.mean_air,data.hi)
    checkpoint=weights_dir/f"epoch{source['selected_epoch']}_m{member}.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source['checkpoint_hashes'][checkpoint.name]
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    model.eval();models.append(model)
cache=precompute(models,data,days,16,row_sum_bound=cap)
residual=np.load(weights_dir/'residual_bank.npz')['residual']
noise=noise_paths(residual,days);bias=np.array(source['bias']);scale=np.array(source['scale'])
traj=simulate(cache,bias,noise,scale);truth=cache['true'][:,1:]
score=summarize(traj,cache['true'],scoring_sd)
np.testing.assert_allclose(score['normalized_mae'],source['validation']['normalized_mae'],rtol=1e-12)
references={}
original=json.loads((ROOT/'Lee/experiments/e16/summary.json').read_text())
for name in ['old','persistence']:
    os.environ['DARROW_FAST_DYNAMICS']='0' if name=='persistence' else '1'
    path=Path(original['model_paths'][name]) if name=='old' else a.runs.parent/'e16_cooperative/persistence.pkl'
    reference_traj=recursive(GBMDynamics(path),arrays(raw['data'],days))
    references[name]=summarize(reference_traj[:,None],cache['true'],scoring_sd)

# Independent Torch forward and cached NumPy rollout must agree for a full day.
probe=precompute(models,data,days[:1],16,row_sum_bound=cap);positions=data.positions[days[0]][:95][None]
exo=torch.tensor(data.exohistory(positions,16),dtype=torch.float32)
context=torch.tensor(data.past(positions[:,0],16),dtype=torch.float32)
state=torch.tensor(data.state[positions[:,0]],dtype=torch.float32)
air=torch.tensor(np.log1p(np.concatenate([probe['zone'],probe['previous_zone']],-1)/1000),dtype=torch.float32)
prediction=simulate(probe,np.zeros(5))
with torch.no_grad():
    for member,model in enumerate(models):
        np.testing.assert_allclose(prediction[member,0],model(state,context,exo,air).numpy(),rtol=2e-4,atol=2e-4)

mae=np.abs(traj-truth[None,None]).mean((0,1,3))
daily={'normalized_mae':(mae/scoring_sd).mean(1),'nh4_mae':mae[:,0],
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
        previous=s
        s=np.einsum('mdjk,mrdk->mrdj',cache['V'][:,:,t],s)+f[:,None,:,t]+np.r_[bias,0,0]
        s=apply_reaction(cache,previous,s,bias,t)
        s[...,:5]+=innov[:,:,:,t]*scale;s=np.clip(s,0,cache['hi']);values.append(s[...,:5].copy())
    reset[str(horizon)]=summarize(np.stack(values,3),cache['true'],scoring_sd)
np.testing.assert_allclose(reset['95']['normalized_mae'],score['normalized_mae'],rtol=1e-8)
conversion=models[0].sd.numpy()*models[0].sign.numpy()
transformed=cache['V']*conversion[None,None,None,None,:]/conversion[None,None,None,:,None]
row_sums=transformed.sum(-1)
result={'selected_epoch':source['selected_epoch'],'selected_scale':source['selected_scale'],
    'model_kind':a.model_kind,'scoring_sd':scoring_sd.tolist(),
    'matrix_row_sum_max':float(row_sums.max()),'matrix_rows_above_one':int((row_sums>1).sum()),
    'event_diagnostics':event_diagnostics(traj,cache['true']),
    'torch_numpy_agreement_passed':True,'reconstruction_passed':True,'test_evaluated':False,
    'uncertainty_unit':'dates, circular five-day blocks, exploratory reused validation, not policy seeds',
    'date_intervals':intervals,'reset_diagnostic':reset,'daily':{k:v.tolist() for k,v in daily.items()},
    'reference_scores':references}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(intervals),flush=True)
