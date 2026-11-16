"""Compare frozen factual event ranking, onsets and date-paired Brier errors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e42'))
from quantile_env import FrozenDynamics,WORKSPACE
from quantile_dynamics import arrays,paths,rollout
from cooperative import recursive,GBMDynamics
from event_timing import onset_counts

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
dyn=FrozenDynamics(a.freeze);raw=pd.read_pickle(a.data);days=raw['days']['val'];batch=arrays(raw['data'],days)
noise=paths(dyn.bank,days,95);new=rollout(dyn.model,batch,noise,.5)
old=recursive(GBMDynamics(WORKSPACE/dyn.manifest['original_environment_path']),batch)[:,None]
truth=batch[0];persistence=np.broadcast_to(truth[None,None,:,0,None,:],(1,1,len(days),95,5))
summary=json.loads((dyn.path/'summary.json').read_text());reference=json.loads((ROOT/'Lee/experiments/e40/audit.json').read_text())
rng=np.random.default_rng(20261112);starts=rng.integers(0,len(days),size=(10000,8))
indices=((starts[:,:,None]+np.arange(5))%len(days)).reshape(10000,-1)[:,:len(days)]
result={'test_evaluated':False,'model_refit':False,'selection_changed':False,
    'scope':'Exploratory factual validation only. Threshold.5 and onset tolerances0/1/4 steps are diagnostics, not selection criteria.',
    'onset_definition':'New false-to-true transitions, excluding ongoing initial violations. Maximum-cardinality one-to-one matching then minimum absolute delay, within each daily episode.',
    'metrics':{}}
errors={}
for name,trajectory in [('old',old),('persistence',persistence),('e40',new)]:
    row={};errors[name]={}
    for label,j,limit in [('nh4',0,4),('do',None,2)]:
        events=trajectory[...,0]>4 if j==0 else trajectory[...,2:].max(-1)>2
        target=truth[:,1:,0]>4 if j==0 else truth[:,1:,2:].max(-1)>2
        initial=truth[:,0,0]>4 if j==0 else truth[:,0,2:].max(-1)>2
        probability=events.mean((0,1));positive=probability>=.5
        error=((probability-target)**2).mean(1);errors[name][label]=error
        expected=(summary['validation'][summary['selected']][f'{label}_brier'] if name=='e40'
            else reference['reference_scores'][name][f'{label}_brier'])
        np.testing.assert_allclose(error.mean(),expected,rtol=1e-12)
        onsets={}
        for tolerance in (0,1,4):
            counts=np.sum([onset_counts(y,p,i,tolerance) for y,p,i in zip(target,positive,initial)],axis=0)
            actual_count,predicted_count,matched,delay=counts
            onsets[str(tolerance)]={'actual_onsets':int(actual_count),'predicted_onsets':int(predicted_count),
                'matched':int(matched),'recall':float(matched/actual_count) if actual_count else None,
                'precision':float(matched/predicted_count) if predicted_count else None,
                'mean_absolute_delay_steps':float(delay/matched) if matched else None}
        tp=int((target&positive).sum());fp=int((~target&positive).sum());fn=int((target&~positive).sum())
        row[label]={'prevalence':float(target.mean()),'brier':float(error.mean()),
            'average_precision':float(average_precision_score(target.ravel(),probability.ravel())),
            'roc_auc':float(roc_auc_score(target.ravel(),probability.ravel())),
            'precision_at_half':tp/(tp+fp) if tp+fp else None,'recall_at_half':tp/(tp+fn) if tp+fn else None,
            'onset_tolerance_steps':onsets}
    result['metrics'][name]=row
result['paired_date_brier_differences']={}
for baseline in ('old','persistence'):
    result['paired_date_brier_differences'][baseline]={}
    for label in ('nh4','do'):
        difference=errors['e40'][label]-errors[baseline][label]
        result['paired_date_brier_differences'][baseline][label]={'mean':float(difference.mean()),
            'circular_five_day_bootstrap95_ci':np.quantile(difference[indices].mean(1),[.025,.975]).tolist()}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['paired_date_brier_differences']),flush=True)
