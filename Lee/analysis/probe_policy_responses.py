"""Probe nearby training-observation responses before proposing actor priors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e44'))
from evaluate_pair import load_run,file_hashes
from surrogate_dw import STATE

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
torch.set_num_threads(1);freeze=json.loads(a.freeze.read_text())
expected=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected
raw=pd.read_pickle(a.data);blocks=[]
for day in raw['days']['train']:
    g=raw['data'].loc[day];air=g.air.to_numpy(float);previous=np.r_[air[:1],air[:-2]]
    blocks.append(np.column_stack([g[STATE].to_numpy(float)[:-1],g.q_in.to_numpy()[:-1]/1000,
        g.temp.to_numpy()[:-1]/10,g.sin.to_numpy()[:-1],g.cos.to_numpy()[:-1],
        g.inf_s_nh.to_numpy()[:-1]/50,previous/9000]))
all_obs=np.concatenate(blocks).astype(np.float32)
indices=np.linspace(0,len(all_obs)-1,min(1024,len(all_obs)),dtype=int);observations=all_obs[indices]
result={'training_observations_only':True,'probe_rows':len(indices),'test_evaluated':False,'policy_refit':False,
    'scope':'Counterfactual nearby observation responses, not measured process effects, causal attribution or an optimal-policy monotonicity theorem. No actor regularizer is fitted.','policies':[]}
def stats(values):
    return {'mean_change':float(values.mean()),'negative_fraction':float((values<-1e-5).mean()),
        'positive_fraction':float((values>1e-5).mean()),'near_zero_fraction':float((np.abs(values)<=1e-5).mean())}
for folder in a.runs:
    agent,config,metrics,hashes=load_run(folder,a.freeze)
    @torch.no_grad()
    def predict(obs):
        normalized=agent._t(agent.obs_normalizer(obs))
        raw_action=agent.pi(agent.policy_input(normalized)).cpu().numpy()[:,0]
        return raw_action,np.clip(raw_action,-1,1)
    baseline_raw,baseline=predict(observations);probes={}
    for label,column,delta in [('nh4_plus_half',0,.5),('do1_plus_point2',2,.2),('do2_plus_point2',3,.2),('do3_plus_point2',4,.2)]:
        changed=observations.copy();changed[:,column]+=delta
        higher_raw,higher=predict(changed)
        probes[label]={'raw_mean':stats(higher_raw-baseline_raw),'executed_action':stats(higher-baseline)}
    assert file_hashes(folder/'model_final')==hashes
    result['policies'].append({'method':config['method'],'policy_seed':config['seed'],
        'training_environment':config['experiment_metadata']['environment'],
        'native_zero_action_fraction_on_probe':float((baseline==-1).mean()),'responses':probes})
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print('Training-only policy input probes complete:',len(result['policies']),flush=True)
