"""Describe action saturation/allocation without changing a failed comparison."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e44'))
from evaluate_pair import load_run,FrozenDynamics,file_hashes,frozen_digest
from quantile_env import FrozenQuantileEnv

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--pair',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
torch.set_num_threads(1);pair=json.loads(a.pair.read_text())
assert pair['evaluation_environment']=='new' and pair['training_steps']==401280
assert pair['freeze_canonical_sha256']==frozen_digest(a.freeze)
dyn=FrozenDynamics(a.freeze);days=pd.read_pickle(a.data)['days']['val']
cases=[(d,m,r) for m in range(5) for r in range(8) for d in days]
result={'policy_seed':pair['policy_seed'],'test_evaluated':False,'policy_refit':False,'matching_rule_changed':False,
    'scope':'Descriptive action diagnostics, not causal attribution or a five-seed performance conclusion.','policies':{}}
for label,folder,offset,expected in [('baseline',a.baseline,0.,pair['baseline_native']),
        ('candidate_matched',a.candidate,pair['matching']['offset'],pair['candidate_matched'])]:
    agent,config,metrics,hashes=load_run(folder,a.freeze)
    assert hashes==pair['baseline_checkpoint_hashes' if label=='baseline' else 'candidate_checkpoint_hashes']
    env=FrozenQuantileEnv(a.data,dyn,split='val',n_env=len(cases),seed=0)
    env.fixed=[(d,m) for d,m,r in cases];env.fixed_draws=[r for d,m,r in cases];obs=env.reset()
    total_air=0.;cost=0.;nh4=0.;do=0.;zero=0;maximum=0;clipped=0;absolute_change=0.;previous=None
    conditional={k:[0.,0] for k in ('nh4_high','nh4_low','do_high','do_low')}
    for t in range(95):
        nmask=env.state[:,0]>4;dmask=env.state[:,2:].max(-1)>2
        raw=agent.act_det(obs)[:,0];shifted=raw+offset
        executed=np.clip(np.asarray(shifted,float),-1,1)
        clipped+=int(((shifted<-1)|(shifted>1)).sum());zero+=int((executed==-1).sum());maximum+=int((executed==1).sum())
        obs,reward,step_cost,done,terminal,info=env.step(shifted)
        air=info['air'];total_air+=float(air.sum()*.25);cost+=float(step_cost.sum())
        nh4+=float(info['v_nh4'].sum());do+=float(info['v_do'].sum())
        if previous is not None:absolute_change+=float(np.abs(air-previous).sum())
        previous=air.copy()
        for key,mask in [('nh4_high',nmask),('nh4_low',~nmask),('do_high',dmask),('do_low',~dmask)]:
            conditional[key][0]+=float(air[mask].sum());conditional[key][1]+=int(mask.sum())
    count=len(cases);steps=95*count
    score={'air_m3':total_air/count,'cost':cost/count,'nh4_viol_pct':nh4/steps*100,'do_viol_pct':do/steps*100}
    for key,value in score.items():np.testing.assert_allclose(value,expected[key],rtol=1e-10,atol=1e-10)
    assert file_hashes(folder/'model_final')==hashes
    result['policies'][label]={'offset':offset,'replayed_metrics':score,'zero_air_fraction':zero/steps,
        'maximum_air_fraction':maximum/steps,'offset_clipped_fraction':clipped/steps,
        'mean_absolute_air_change_m3_per_hour':absolute_change/(94*count),
        'conditional_mean_air_m3_per_hour':{k:total/n if n else None for k,(total,n) in conditional.items()}}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['policies']),flush=True)
