"""Day-block uncertainty and severity diagnostics for the frozen e16 candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cooperative import arrays,recursive,GBMDynamics,STATE

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
summary=json.loads((a.runs/'summary.json').read_text())
data=pd.read_pickle(a.data)
days=data['days']['val']
batch=arrays(data['data'],days)
true=batch[0][:,1:]
fit=np.concatenate([data['data'].loc[day][STATE].to_numpy(float) for day in summary['fit_days']])
sd=fit.std(0)
train_min_temp=min(float(data['data'].loc[day].temp.min()) for day in data['days']['train'])
day_cold=np.array([float(data['data'].loc[day].temp.min())<train_min_temp for day in days])
paths={k:Path(summary['model_paths'][k]) for k in ['old','direct7',summary['selected_by_inner_calibration']]}
paths['persistence']=a.runs/'persistence.pkl'
metrics={}
for name,path in paths.items():
    os.environ['DARROW_FAST_DYNAMICS']='0' if name=='persistence' else '1'
    traj=recursive(GBMDynamics(path),batch)
    per_day_mae=np.abs(traj-true[None]).mean((0,2))
    per_day_n=(traj[...,0]>4).mean((0,2))*100
    per_day_d=(traj[...,2:].max(-1)>2).mean((0,2))*100
    severity_n=np.maximum(traj[...,0]-4,0).sum(2).mean(0)*.25
    severity_d=np.maximum(traj[...,2:].max(-1)-2,0).sum(2).mean(0)*.25
    metrics[name]={'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'per_day_mae':per_day_mae.tolist(),'per_day_normalized_mae':(per_day_mae/sd).mean(1).tolist(),
        'per_day_nh4_violation_pct':per_day_n.tolist(),'per_day_do_violation_pct':per_day_d.tolist(),
        'nh4_excess_mg_h_L_mean':float(severity_n.mean()),'do_excess_mg_h_L_mean':float(severity_d.mean()),
        'worst_day_nh4_violation_pct':float(per_day_n.max()),'worst_day_do_violation_pct':float(per_day_d.max()),
        'temperature_groups':{}}
    for group,mask in [('below_train_range_day',day_cold),('other_day',~day_cold)]:
        if not mask.any():continue
        metrics[name]['temperature_groups'][group]={'days':int(mask.sum()),
            'nh4_mae':float(per_day_mae[mask,0].mean()),
            'nh4_pred_pct':float(per_day_n[mask].mean()),'nh4_actual_pct':float((true[mask,:,0]>4).mean()*100),
            'do_pred_pct':float(per_day_d[mask].mean()),'do_actual_pct':float((true[mask,:,2:].max(-1)>2).mean()*100)}
rng=np.random.default_rng(20261112)
starts=rng.integers(0,len(days),size=(10000,int(np.ceil(len(days)/5))))
indices=((starts[:,:,None]+np.arange(5))%len(days)).reshape(10000,-1)[:,:len(days)]
intervals={}
for name in paths:
    if name=='old':continue
    intervals[name]={}
    for key in ['per_day_normalized_mae','per_day_nh4_violation_pct','per_day_do_violation_pct']:
        diff=np.asarray(metrics[name][key])-np.asarray(metrics['old'][key])
        intervals[name][key]={'mean_difference':float(diff.mean()),
            'five_day_block_bootstrap_95_ci':np.quantile(diff[indices].mean(1),[.025,.975]).tolist()}
result={'days':days,'selected':summary['selected_by_inner_calibration'],'test_evaluated':False,
        'uncertainty_unit':'validation date, circular five-day moving blocks, not policy training seeds',
        'metrics':metrics,'paired_vs_old':intervals}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({'selected':result['selected'],'paired_vs_old':intervals}),flush=True)
