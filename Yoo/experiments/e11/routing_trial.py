"""유입 및 유출 반응조 신호 누락을 같은 예산의 새 모델로 대조한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'e10'))
from history_trial import Model, batches, arrays, pack, event_metrics, write, CONFIG
from audit_inputs import COLUMNS


def routing_frame(path):
    columns=[COLUMNS[k] for k in ('inlet','outlet')]
    raw=pd.read_csv(path,usecols=['time']+[c+s for c in columns for s in (' value',' quality')])
    raw.index=pd.to_datetime(raw.pop('time'),utc=True).dt.tz_convert('Europe/Copenhagen')
    raw=raw.loc['2022-09-15':'2023-12-31']
    features={}
    for name,col in zip(('inlet','outlet'),columns):
        value=raw[col+' value'].where(raw[col+' quality'].eq(0))
        for code in (0,1,2):
            features[f'{name}_{code}']=(value==code).astype(float).where(value.notna()).resample('5min').mean()
    frame=pd.DataFrame(features)
    for name in ('inlet','outlet'):
        frame[name+'_known']=frame[name+'_0'].notna().astype(float)
    frame=frame.fillna(0.)
    for name in ('inlet','outlet'):
        np.testing.assert_allclose(frame[[name+'_'+str(k) for k in (0,1,2)]].sum(1),frame[name+'_known'])
    return frame


@torch.no_grad()
def replay(model,x,aux,mode,reset):
    state,exog=pack(x)
    state,exog=torch.tensor(state).float(),torch.tensor(exog).float()
    extra=torch.tensor(aux).float()
    if mode=='without_routing':extra=torch.zeros_like(extra)
    pred=state[:,12];result=[]
    for t in range(12,287):
        if (t-12)%reset==0:pred=state[:,t]
        pred=model(pred,exog[:,t],extra[:,t].reshape(-1,4,2))
        assert torch.isfinite(pred).all()
        result.append(pred.numpy())
    return np.stack(result,1)


def finalize(report):
    for seed in CONFIG['seeds']:
        pair=[r for r in report['training'] if r['seed']==seed]
        for k in ('initial_parameter_sha256','batch_sha256','parameter_count'):assert pair[0][k]==pair[1][k]
    checks=[]
    for seed in CONFIG['seeds']:
        def row(m,h):return next(r for r in report['evaluation'] if r['mode']==m and r['seed']==seed and r['split']=='inner_holdout' and r['reset_steps']==h)
        c,h=row('without_routing',275),row('with_routing',275)
        c1,h1=row('without_routing',1),row('with_routing',1)
        checks.append({'seed':seed,'cost_mae_improved':h['cost']['daily_mae']<c['cost']['daily_mae'],
            'onset_f1_not_worse':h['nh4']['onset_f1']>=c['nh4']['onset_f1'],
            'longest_run_mae_not_worse':h['nh4']['longest_observed_run_mae_minutes']<=c['nh4']['longest_observed_run_mae_minutes'],
            'do_mae_within_5pct':h['do']['mae']<=c['do']['mae']*1.05,
            'one_step_nh4_mae_within_5pct':h1['nh4']['mae']<=c1['nh4']['mae']*1.05})
    means={mode:np.mean([r['cost']['daily_mae'] for r in report['evaluation'] if r['split']=='inner_holdout'
                        and r['reset_steps']==275 and r['mode']==mode]) for mode in ('without_routing','with_routing')}
    report['gate']={'per_seed':checks,'mean_cost_mae_improvement_fraction':1-means['with_routing']/means['without_routing']}
    report['gate']['passed']=bool(all(v for r in checks for k,v in r.items() if k!='seed') and report['gate']['mean_cost_mae_improvement_fraction']>=.05)
    report['checks']={'paired_initial_parameters':True,'paired_batches':True,'finite_training':True,
                      'save_restore_exact':True,'future_state_not_used':True}
    return report


def main(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.finalize_only:
        if (a.out/'results.json').exists():raise ValueError('Results already exist')
        report=json.loads((a.out/'progress.json').read_text())
        assert len(report['training'])==4 and len(report['evaluation'])==24
        source=subprocess.check_output(['git','show',report['commit']+':Yoo/experiments/e11/routing_trial.py'])
        assert hashlib.sha256(source).hexdigest()==report['source_sha256']
        for row in report['training']:
            path=a.out/f"{row['mode']}_s{row['seed']}.pt"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['model_sha256']
        for row in report['evaluation']:
            z=np.load(a.out/f"{row['mode']}_s{row['seed']}_{row['split']}_h{row['reset_steps']}.npz")
            expected=event_metrics(z['pred'],z['truth'],z['initial'])
            assert all(row[k]==v for k,v in expected.items())
        finalize(report)
        report['finalizer_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        report['recovered_from']='JSON bool serialization failure after all training and evaluation'
        report['completed_at']=datetime.now(timezone.utc).isoformat()
        write(a.out/'results.json',report)
        print(json.dumps(report['gate']),flush=True)
        return
    if a.out.exists() and any(a.out.iterdir()):raise ValueError(f'Refusing overwrite {a.out}')
    a.out.mkdir(parents=True)
    data=pd.read_pickle(a.data);days=data['days']['train'];assert len(days)==330
    partition={'fit':days[:263],'purge':days[263:264],'inner_holdout':days[264:]}
    temp={'data':data['data'],'days':partition}
    xs={k:arrays(temp,k) for k in ('fit','inner_holdout')}
    frame=routing_frame(a.raw)
    aux={k:np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in partition[k]]) for k in xs}
    assert all(np.isfinite(x).all() for x in aux.values())
    for split in xs:
        for d in partition[split]:
            idx=data['data'].loc[d].index
            assert len(idx)==288 and idx[0].hour==idx[0].minute==0 and np.all(np.diff(idx.asi8)==300*10**9)
    report={'started_at':datetime.now(timezone.utc).isoformat(),'config':CONFIG|{'lags':[],'extra_inputs':list(frame.columns)},
        'partition':partition,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.raw,a.data)},
        'test_used':False,'outer_val_used':False,'inner_holdout_reused_after_e10':True,
        'versions':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},'training':[],'evaluation':[]}
    for seed in CONFIG['seeds']:
        for mode in ('without_routing','with_routing'):
            torch.manual_seed(seed)
            batch=list(batches(xs['fit'],'current_repeat'))
            extra=torch.tensor(aux['fit'][:,12:-1].reshape(-1,4,2)).float()
            batch[2]=extra if mode=='with_routing' else torch.zeros_like(extra)
            model=Model(batch)
            initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
            opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
            rng=np.random.default_rng(10000+seed);digest=hashlib.sha256();logs=[]
            for epoch in tqdm(range(CONFIG['epochs']),desc=f'{mode} seed {seed}',mininterval=5):
                perm=rng.permutation(len(batch[0]));digest.update(perm.tobytes());losses=[]
                for start in range(0,len(perm),CONFIG['batch_size']):
                    ids=perm[start:start+CONFIG['batch_size']]
                    state,exog,extra,target=[v[ids] for v in batch]
                    loss=((model(state,exog,extra)-target)/model.delta_sd).square().mean()
                    opt.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['grad_clip'],error_if_nonfinite=True)
                    assert torch.isfinite(loss)
                    opt.step();losses.append(loss.item())
                logs.append(float(np.mean(losses)))
            path=a.out/f'{mode}_s{seed}.pt';torch.save(model.state_dict(),path)
            restored=Model(batch);restored.load_state_dict(torch.load(path,weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model(*[v[:32] for v in batch[:3]]),restored(*[v[:32] for v in batch[:3]]),rtol=0,atol=0)
            report['training'].append({'mode':mode,'seed':seed,'initial_parameter_sha256':initial_hash,
                'batch_sha256':digest.hexdigest(),'parameter_count':sum(p.numel() for p in model.parameters()),
                'epoch_losses':logs,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for split,x in xs.items():
                truth=np.stack((x['nh4'][:,13:],x['do'][:,13:]),-1)
                initial=np.stack((x['nh4'][:,12],x['do'][:,12]),-1)
                for h in (1,12,275):
                    pred=replay(restored,x,aux[split],mode,h)
                    np.savez_compressed(a.out/f'{mode}_s{seed}_{split}_h{h}.npz',pred=pred,truth=truth,initial=initial)
                    report['evaluation'].append({'mode':mode,'seed':seed,'split':split,'reset_steps':h,
                        'days':len(truth),'evaluated_steps_per_day':275,**event_metrics(pred,truth,initial)})
                # 실제 학습망으로 미래 실측값 비참조를 확인한다.
                tiny={k:v[:2].copy() for k,v in x.items()}
                original=replay(restored,tiny,aux[split][:2],mode,275)
                tiny['nh4'][:,13:]+=1000;tiny['do'][:,13:]+=1000
                np.testing.assert_array_equal(original,replay(restored,tiny,aux[split][:2],mode,275))
            write(a.out/'progress.json',report)
    finalize(report)
    report['completed_at']=datetime.now(timezone.utc).isoformat();write(a.out/'results.json',report)
    print(json.dumps(report['gate']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('raw','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--finalize-only',action='store_true')
    main(p.parse_args())
