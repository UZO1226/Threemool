"""Check a prospective auxiliary label on training observations only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
from surrogate_dw import STATE

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
freeze=json.loads(a.freeze.read_text())
expected=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected
data=pd.read_pickle(a.data)
environment_summary=ROOT.parent/freeze['environment_run']/'summary.json'
summary_key=next(k for k in freeze['frozen_files_sha256'] if k.endswith('summary.json'))
assert hashlib.sha256(environment_summary.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==freeze['frozen_files_sha256'][summary_key]
fit_days=json.loads(environment_summary.read_text())['fit_days']
assert set(fit_days).issubset(set(data['days']['train']))
fit_states=np.concatenate([data['data'].loc[d][STATE].to_numpy(float) for d in fit_days])
clip_hi=fit_states.max(0)*1.1
maximum_new_label=2+max(clip_hi[0]/4-1,0)+max(clip_hi[2:].max()/2-1,0)
distribution_high=float(np.expm1(4))
assert maximum_new_label<distribution_high
states=np.concatenate([data['data'].loc[d][STATE].to_numpy(float)[1:] for d in data['days']['train']])
assert states.shape==(188*95,5) and np.isfinite(states).all()
nh4=states[:,0];do=states[:,2:].max(-1)
cost=(nh4>4).astype(float)+(do>2).astype(float)
severity=np.maximum(nh4/4-1,0)+np.maximum(do/2-1,0)
aux=cost+severity
assert np.array_equal(cost==0,aux==0)
rows=[]
for c in (0,1,2):
    values=aux[cost==c]
    rows.append({'actual_cost':c,'count':int(len(values)),
        'auxiliary_quantiles':np.quantile(values,[0,.25,.5,.75,1]).tolist() if len(values) else None})
out={'training_only':True,'training_days':len(data['days']['train']),'transitions':len(states),
    'test_evaluated':False,'validation_evaluated':False,'policy_training':False,
    'label':'c + max(NH4/4-1,0) + max(maxDO/2-1,0)',
    'same_zero_cost_set':True,'actual_cost_groups':rows,'maximum_auxiliary':float(aux.max()),
    'environment_clip_hi':clip_hi.tolist(),'maximum_new_state_label_bound':float(maximum_new_label),
    'distribution_original_scale_high':distribution_high,
    'discounted_max_target_upper_bound':float(max(maximum_new_label,.9*distribution_high)),
    'labels_within_existing_distribution_support':True,
    'scope':'Factual training distribution and conservative clipped-state label bound. No policy improvement or causal effect claim. Clip bounds reconstructed from the verified environment fit days and existing1.1 rule.'}
a.out.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out),flush=True)
