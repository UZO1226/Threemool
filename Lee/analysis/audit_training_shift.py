"""Describe observed split shift and train-only historical curriculum weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
from curriculum_dw import risk_day_weights

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
freeze=json.loads(a.freeze.read_text())
expected=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected
data=pd.read_pickle(a.data);table=data['data'];weights,rule=risk_day_weights(table,data['days']['train'])
def daily(days):
    rows=[]
    for day in days:
        g=table.loc[day];nh4=g.nh4.to_numpy(float);do=g[['do1','do2','do3']].to_numpy(float).max(-1)
        rows.append([np.mean(nh4[1:]>4)*100,np.mean(do[1:]>2)*100,float(nh4[0]>4),
            float(do[0]>2),nh4.mean(),do.mean(),g.q_in.mean(),g.temp.mean()])
    return np.asarray(rows)
keys=['nh4_viol_pct','do_viol_pct','initial_nh4_violation_fraction','initial_do_violation_fraction',
    'nh4_mean','maximum_zone_do_mean','inflow_mean','temperature_mean']
train=daily(data['days']['train']);val=daily(data['days']['val'])
cold_count=int(np.ceil(len(train)*.25));cold=np.argsort(train[:,-1],kind='stable')[:cold_count]
cold_weights=np.full(len(train),.5/(len(train)-cold_count));cold_weights[cold]=.5/cold_count
policy=[]
for seed in range(5):
    folder=a.baseline_root/f'seed_{seed}'
    progress=json.loads((folder/'progress.json').read_text());assert progress['phase']=='complete'
    log=json.loads((folder/'train_log.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    assert metrics['steps']==401280
    policy.append({'seed':seed,'last10_training_rollout_cost_mean':float(np.mean([row['train_episode_cost'] for row in log[-10:]])),
        'final_uniform_validation_cost':metrics['val_final']['cost']})
out={'test_evaluated':False,'validation_used_for_diagnosis':True,'policy_refit':False,
    'sampling_rule_modified':False,'historical_rule_source':'Lee/code/darrow/curriculum_dw.py, used in e15',
    'training_days':len(train),'validation_days':len(val),
    'observed_uniform_train':dict(zip(keys,train.mean(0).tolist())),
    'observed_uniform_validation':dict(zip(keys,val.mean(0).tolist())),
    'observed_curriculum_weighted_train':dict(zip(keys,(train*weights[:,None]).sum(0).tolist())),
    'observed_cold_weighted_train':dict(zip(keys,(train*cold_weights[:,None]).sum(0).tolist())),
    'cold_rule':'Coldest training-day temperature quarter receives half probability; rest half. Stable date-order ties. No validation-selected temperature cutoff.',
    'sampling_weight_range':[float(weights.min()),float(weights.max())],
    'sampling_effective_number_of_days':float(1/(weights@weights)),'baseline_policy_diagnostics':policy,
    'limitations':'Observed split averages and changing-policy rollout costs are descriptive, not causal or independent confirmation. Weighting changes the training/PID distribution. The unchanged e15 rule previously failed on another simulator; no retry success is assumed.'}
a.out.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out),flush=True)
