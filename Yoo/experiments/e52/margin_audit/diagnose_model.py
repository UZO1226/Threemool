"""원 기록 상태 한 스텝과 기록 행동 반복 예측의 NH4 오차를 같은 분할에서 진단한다."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'e45'))
from evaluate import load_selected
from darrow_env import Dynamics
from common import sha,save

RESULTS=Path('/home/jio/fcsrl_src/results')


def summarize(pred,actual,sd):
    assert pred.shape==actual.shape==sd.shape
    safe=actual<=4;bad=actual>4;flag=pred>4
    fp=safe&flag;fn=bad&~flag
    err=pred-actual
    corr=lambda target:float(pd.Series(sd.ravel()).corr(pd.Series(target.ravel()),method='spearman'))
    return dict(predicted_cost=float(flag.sum(-1).mean()),actual_cost=float(bad.sum(-1).mean()),
                false_positive_rate_pct=float(100*fp.sum()/safe.sum()),recall_pct=float(100*(flag&bad).sum()/bad.sum()),
                mae=float(abs(err).mean()),signed_error=float(err.mean()),underprediction_mae=float(np.maximum(-err,0).mean()),
                overprediction_mae=float(np.maximum(err,0).mean()),
                sd_absolute_error_spearman=corr(abs(err)),sd_underprediction_spearman=corr(np.maximum(-err,0)),
                counts=dict(false_positive=int(fp.sum()),false_negative=int(fn.sum()),actual_bad=int(bad.sum())))


def run(a):
    source=RESULTS/'e30/prepared/data.pkl';model=RESULTS/'e35/zone_fix/corrected_surrogate.pkl'
    splits,days=load_selected(source);dyn=Dynamics(model)
    out=dict(source_sha256={str(p):sha(p) for p in [source,model,Path(__file__)]},test_evaluated=False,weight_updates=0,
             purpose='posthoc diagnostic, not policy or hyperparameter selection',splits={})
    for split in ['train','val']:
        xs=[];actual=[];current=[];all_air=[]
        for day in splits[split]:
            z=days[day];zone=z['zone'][:-1];prev=np.concatenate([z['zone'][:1],z['zone'][:-2]],axis=0)
            xs.append(np.column_stack([z['state'][:-1],z['exo'][:-1],zone/1000,prev/1000]))
            actual.append(z['state'][1:,0]);current.append(z['state'][:-1,0]);all_air.append(z['air'][:-1])
        x=np.concatenate(xs);current=np.concatenate(current)
        delta=np.stack([m[0].predict(x) for m in dyn.models])
        sd=delta.std(0).reshape(len(actual),95)
        pred=np.clip(current[None,:]+delta,0,dyn.hi[0]).reshape(5,len(actual),95).transpose(1,0,2)
        truth=np.repeat(np.stack(actual)[:,None,:],5,axis=1)
        repeated_sd=np.repeat(sd[:,None,:],5,axis=1)
        with np.load(a.calibration/f'{split}_recorded_replay.npz') as z:
            rollout=z['nh4'].reshape(-1,5,95);rsd=z['sd'].reshape(-1,5,95)
        alarm_stats={}
        for name,margin in [('fixed',.45),('adaptive',10*rsd)]:
            flag=rollout+margin>4;bad=truth>4;added=flag&~(rollout>4)
            alarm_stats[name]=dict(cost=float(flag.sum(-1).mean()),
                false_positive_rate_pct=float(100*(flag&~bad).sum()/(~bad).sum()),
                recall_pct=float(100*(flag&bad).sum()/bad.sum()),
                added_true_alarms=int((added&bad).sum()),added_false_alarms=int((added&~bad).sum()),
                added_alarm_precision_pct=float(100*(added&bad).sum()/max(1,added.sum())))
        air=np.concatenate(all_air)
        out['splits'][split]=dict(n_days=len(actual),recorded_action_clipping_pct=float(100*np.mean((air<0)|(air>9000))),
                                  observed_state_one_step=summarize(pred,truth,repeated_sd),
                                  recursive_recorded_action=summarize(rollout,truth,rsd),
                                  margin_alarm_stats=alarm_stats)
        print(split,json.dumps(out['splits'][split],ensure_ascii=False),flush=True)
    assert all(sha(p)==v for p,v in out['source_sha256'].items())
    save(a.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
