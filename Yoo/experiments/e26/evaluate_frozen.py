"""고정 e14 모델의 시간 구간 전이를 평가한다. 재학습과 test 평가는 없다."""
import argparse
from datetime import datetime, timezone
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e12'))
from trial import load_model,routing_frame,arrays,replay,score,gates
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise ValueError('Refusing overwrite')
a.out.mkdir(parents=True)
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
root=Path('/home/jio/fcsrl_src/results')
data_path=HERE.parents[1]/'data/avedore/aved_5min.pkl'
raw_path=data_path.with_name('aved_raw.csv')
data=pd.read_pickle(data_path);days=data['days']['val'];assert len(days)==43
assert all('2024-01-01'<=d<='2024-02-29' for d in days)
x=arrays(data,'val');frame=routing_frame(raw_path,end='2024-02-29')
aux=np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in days]);assert np.isfinite(aux).all()
paths={mode:[root/folder/f'{name}_s{s}.pt' for s in (0,1)] for mode,folder,name in [
 ('reference','e11/routing_trial','with_routing'),('single_mse','e12/trial','single_mse'),
 ('start_multi','e12/trial','multi_mse'),('short_mse','e14/trial','short_mse'),('day_mse','e14/trial','day_mse')]}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[data_path,raw_path,*[p for ps in paths.values() for p in ps]]
sources=[Path(__file__),*[HERE.parent/f'e{n}/{name}' for n,name in [(12,'trial.py'),(11,'routing_trial.py'),(10,'history_trial.py')]]]
report={'started_at':datetime.now(timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'input_sha256':{str(p):sha(p) for p in inputs},'source_sha256':{str(p):sha(p) for p in sources},'days':days,'test_used':False,'training_performed':False,'evaluation':[]}
for mode,ps in paths.items():
 for seed,path in enumerate(ps):
  model=load_model(path)
  for reset in (1,275):
   pred=replay(model,x,aux,'with_routing',reset)
   report['evaluation'].append({'mode':mode,'seed':seed,'split':'val','reset_steps':reset,**score(pred,x)})
   if reset==275:
    changed={k:v.copy() for k,v in x.items()};changed['nh4'][:,13:]+=1000;changed['do'][:,13:]+=1000
    np.testing.assert_array_equal(pred,replay(model,changed,aux,'with_routing',reset))
  print(mode,seed,'done',flush=True)
rows=report['evaluation']
report['eligibility']=gates(rows,'val','day_mse')
def get(mode,seed):return next(r for r in rows if r['mode']==mode and r['seed']==seed and r['reset_steps']==275)
pairs=[]
for seed in (0,1):
 c,b=get('day_mse',seed),get('start_multi',seed)
 pairs.append({'seed':seed,'recall_not_worse':c['nh4']['step_recall']>=b['nh4']['step_recall'],'fpr_not_worse':c['nh4']['fpr']<=b['nh4']['fpr']})
cost=lambda m:float(np.mean([get(m,s)['cost']['daily_mae'] for s in (0,1)]))
reduction=1-cost('day_mse')/cost('start_multi')
report['partial_effect']={'checks':pairs,'mean_cost_mae_reduction':reduction,'passed':all(r[k] for r in pairs for k in ('recall_not_worse','fpr_not_worse')) and reduction>=.05}
assert all(sha(Path(p))==h for p,h in {**report['input_sha256'],**report['source_sha256']}.items())
report['input_and_source_unchanged']=True;report['future_state_not_used']=True;report['completed_at']=datetime.now(timezone.utc).isoformat()
(a.out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
lines=['# e26 固定 모델의 외부 시간 구간 결과'.replace('固定','고정'),'','val43일, 하루275전이, seed2개 평균. 실제 운전 기록 재생이며 정책 제어 성과가 아니다.','','| 모델 | NH4 recall | NH4 FPR | NH4 MAE | DO recall | DO FPR | 비용 MAE |','| --- | --- | --- | --- | --- | --- | --- |']
for mode in paths:
 rs=[get(mode,s) for s in (0,1)]
 avg=lambda name,key:np.mean([r[name][key] for r in rs])
 lines.append(f"| {mode} | {100*avg('nh4','step_recall'):.2f}% | {100*avg('nh4','fpr'):.2f}% | {avg('nh4','mae'):.3f} | {100*avg('do','step_recall'):.2f}% | {100*avg('do','fpr'):.2f}% | {avg('cost','daily_mae'):.2f} |")
lines+=['',f"부분 효과 재현: {report['partial_effect']['passed']}. 시작 모델 대비 평균 비용 MAE 감소율 {reduction*100:.2f}%.",f"기존 screening: {report['eligibility']['screen_passed']}. 절대 recall80/FPR10: {report['eligibility']['absolute_recall80_fpr10']}. 환경 교체 적격성: {report['eligibility']['pilot_eligibility']}.",'','가중치, 분할, 물리 기준을 변경하지 않았다. test 미사용. 반복 사용한 val의 탐색 결과다. 상세 seed별 지표와 실패 항목은 evaluation.json에 있다.']
(a.out/'results.md').write_text('\n'.join(lines));print('\n'.join(lines))
