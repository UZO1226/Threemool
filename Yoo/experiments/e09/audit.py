"""정책 실패 조건과 대리 모델의 오차 누적을 train 및 val에서 분리한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/wwtp'))
import numpy as np
import pandas as pd
from surrogate import Surrogate, do_inputs, nh4_inputs, phase_onehot


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def arrays(data, split):
    frame = data['data']
    return {k: np.stack([frame.loc[d][k].to_numpy() for d in data['days'][split]])
            for k in ('nh4','do','air','phase','temp','q_in')}


def simulate(sur, x, member, reset=287, oracle=None):
    N, length = x['nh4'].shape
    n, d = x['nh4'][:,0].copy(), x['do'][:,0].copy()
    predicted = []
    for t in range(length-1):
        if t % reset == 0:
            n, d = x['nh4'][:,t].copy(), x['do'][:,t].copy()
        if oracle == 'nh4': n = x['nh4'][:,t].copy()
        if oracle == 'do': d = x['do'][:,t].copy()
        ph = phase_onehot(x['phase'][:,t])
        s, c = np.full(N,np.sin(2*np.pi*t/288)), np.full(N,np.cos(2*np.pi*t/288))
        xd = do_inputs(d,x['air'][:,t],ph,x['temp'][:,t],n,x['q_in'][:,t],s,c)
        dn = np.maximum(0,d+sur.do_model.predict(xd,member)[0])
        if oracle == 'do': dn = x['do'][:,t+1].copy()
        xn = nh4_inputs(n,d,dn,ph,x['temp'][:,t],x['q_in'][:,t],s,c)
        nn = np.maximum(0,n+sur.nh4_model.predict(xn,member)[0])
        n,d = nn,dn
        predicted.append(np.stack((n,d),-1))
    return np.stack(predicted,1)


def replay_report(sur, x, days, details):
    truth = np.stack((x['nh4'][:,1:],x['do'][:,1:]),-1)
    true_cost = (truth[:,:,0]>4).astype(float)+(truth[:,:,1]>2)
    result=[]
    for reset,oracle in [(1,None),(12,None),(36,None),(72,None),(287,None),(287,'do'),(287,'nh4')]:
        for member in range(sur.n_members):
            pred = simulate(sur,x,member,reset,oracle)
            error=pred-truth
            cost=(pred[:,:,0]>4).astype(float)+(pred[:,:,1]>2)
            row={'reset_steps':reset,'oracle':oracle,'member':member,
                 'nh4_mae':float(np.abs(error[:,:,0]).mean()),'nh4_bias':float(error[:,:,0].mean()),
                 'do_mae':float(np.abs(error[:,:,1]).mean()),'do_bias':float(error[:,:,1].mean()),
                 'predicted_cost':float(cost.sum(1).mean()),'actual_cost':float(true_cost.sum(1).mean()),
                 'nh4_false_negative_pct':float(((truth[:,:,0]>4)&(pred[:,:,0]<=4)).mean()*100),
                 'nh4_false_positive_pct':float(((truth[:,:,0]<=4)&(pred[:,:,0]>4)).mean()*100)}
            result.append(row)
            if reset==287 and oracle is None:
                records=[{'day':day,'member':member,'predicted_cost':float(cost[i].sum()),
                    'actual_cost':float(true_cost[i].sum()),'nh4_mae':float(np.abs(error[i,:,0]).mean()),
                    'nh4_bias':float(error[i,:,0].mean())} for i,day in enumerate(days)]
                write(details/f'member_{member}_days.json',records)
    return result


def describe(x):
    return {k:float(v) for k,v in zip(('min','q25','median','q75','max'),np.quantile(x,[0,.25,.5,.75,1]))} | {'mean':float(np.mean(x))}


def audit(a):
    if a.out.exists() and any(a.out.iterdir()): raise ValueError(f'Refusing to overwrite {a.out}')
    a.out.mkdir(parents=True)
    data=pd.read_pickle(a.data)
    x={s:arrays(data,s) for s in ('train','val')}
    temp_floor=float(x['train']['temp'].min())
    report={'started_at':datetime.now(timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.data,a.model)},
        'training_performed':False,'test_used':False,'train_temperature_min':temp_floor,
        'covariates':{},'policy_groups':[],'replay':{}}
    for split,values in x.items():
        features={'initial_nh4':values['nh4'][:,0],'temperature_mean':values['temp'].mean(1),
            'flow_mean':values['q_in'].mean(1),'aerobic_fraction':(values['phase']==2).mean(1)}
        report['covariates'][split]={k:describe(v) for k,v in features.items()}
        report['covariates'][split]['steps_below_train_temperature_min_fraction']=float((values['temp']<temp_floor).mean())
        report['covariates'][split]['days_below_train_temperature_min']=int((values['temp'].min(1)<temp_floor).sum())
        day_groups={day: {'initial_high':bool(values['nh4'][i,0]>4),
            'cold':bool(values['temp'][i].min()<temp_floor)} for i,day in enumerate(data['days'][split])}
        for variant in ('continuous','freeze_half'):
            for seed in (0,1):
                rows=json.loads((a.policy_root/f'{variant}_s{seed}_{split}_episodes.json').read_text())
                frame=pd.DataFrame(rows)
                frame['initial_high']=[day_groups[d]['initial_high'] for d in frame.day]
                frame['cold']=[day_groups[d]['cold'] for d in frame.day]
                groups=[]
                for (member,high,cold),g in frame.groupby(['member','initial_high','cold']):
                    groups.append({'member':int(member),'initial_high':bool(high),'cold':bool(cold),
                        'episodes':len(g),'cost':float(g.cost.mean()),'cost_sum':float(g.cost.sum())})
                report['policy_groups'].append({'variant':variant,'seed':seed,'split':split,
                    'cost':float(frame.cost.mean()),'cold_cost_share':float(frame.loc[frame.cold,'cost'].sum()/frame.cost.sum()),
                    'member0_cost_share':float(frame.loc[frame.member==0,'cost'].sum()/frame.cost.sum()),'groups':groups,
                    'worst_episodes':frame.sort_values('cost',ascending=False).head(10).to_dict('records')})
        directory=a.out/split
        directory.mkdir()
        report['replay'][split]=replay_report(Surrogate(a.model),values,data['days'][split],directory)
        print('DONE',split,flush=True)
        write(a.out/'audit.json',report)
    report['completed_at']=datetime.now(timezone.utc).isoformat()
    write(a.out/'audit.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--policy-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    audit(p.parse_args())
