"""연속 예측 손실과 균형 위반 손실의 효과를 2 x 2 대조한다."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'e11'))
from routing_trial import Model, routing_frame, replay, arrays, pack, event_metrics, write

MODES=('single_mse','single_boundary','multi_mse','multi_boundary')
CFG={'horizon':36,'updates':300,'batch':128,'lr':1e-4,'weight_decay':1e-5,'grad_clip':1.,
     'seeds':[0,1],'sampling_seed_base':12000,'calibration_seed':12999,'boundary_gradient_ratio':.5,
     'thresholds':[4.,2.],'checkpoint':'last','primary_candidate':'multi_boundary'}


def load_model(path):
    model=Model([torch.zeros(2,n) if n!=8 else torch.zeros(2,4,2) for n in (2,6,8,2)])
    model.load_state_dict(torch.load(path,weights_only=True))
    return model


def balanced_boundary(pred,target,prevalence,temperature):
    threshold=pred.new_tensor(CFG['thresholds'])
    label=(target>threshold).to(pred.dtype)
    weights=label/(2*prevalence)+(1-label)/(2*(1-prevalence))
    return (F.binary_cross_entropy_with_logits((pred-threshold)/temperature,label,reduction='none')*weights).mean()


def losses(model,state,exog,extra,scale,prevalence,multi):
    pred=state[:,0];mse=boundary=0.
    for t in range(CFG['horizon']):
        if not multi:pred=state[:,t]
        pred=model(pred,exog[:,t],extra[:,t].reshape(-1,4,2))
        mse=mse+((pred-state[:,t+1])/scale).square().mean()/CFG['horizon']
        boundary=boundary+balanced_boundary(pred,state[:,t+1],prevalence,model.delta_sd)/CFG['horizon']
    return mse,boundary


def window(tensors,rng):
    state,exog,aux=tensors
    days=rng.integers(0,len(state),CFG['batch'])
    starts=rng.integers(12,288-CFG['horizon'],CFG['batch'])
    times=torch.tensor(starts)[:,None]+torch.arange(CFG['horizon']+1)
    ids=torch.tensor(days)[:,None]
    return (state[ids,times],exog[ids,times[:,:-1]],aux[ids,times[:,:-1]]),days.tobytes()+starts.tobytes()


def score(pred,x):
    truth=np.stack((x['nh4'][:,13:],x['do'][:,13:]),-1)
    initial=np.stack((x['nh4'][:,12],x['do'][:,12]),-1)
    metrics=event_metrics(pred,truth,initial)
    for k,name,threshold in ((0,'nh4',4),(1,'do',2)):
        y,p=truth[:,:,k]>threshold,pred[:,:,k]>threshold
        tp,fp,fn,tn=[int(z.sum()) for z in (y&p,~y&p,y&~p,~y&~p)]
        metrics[name].update(tp=tp,fp=fp,fn=fn,tn=tn,fpr=fp/(fp+tn),
            bias=float((pred[:,:,k]-truth[:,:,k]).mean()),
            zero_fraction=float((pred[:,:,k]<=1e-8).mean()),
            zero_on_violation_fraction=float((pred[:,:,k][y]<=1e-8).mean()))
    return metrics


def gates(rows,split,candidate):
    checks=[]
    for seed in CFG['seeds']:
        def get(mode,h):return next(r for r in rows if r['mode']==mode and r['seed']==seed and r['split']==split and r['reset_steps']==h)
        c=get(candidate,275);c1=get(candidate,1)
        for base in ('reference','single_mse'):
            b=get(base,275);b1=get(base,1)
            check={'seed':seed,'base':base,
                'cost_mae_within_5pct':c['cost']['daily_mae']<=b['cost']['daily_mae']*1.05,
                'onset_f1_not_worse':c['nh4']['onset_f1']>=b['nh4']['onset_f1'],
                'longest_run_mae_within_5pct':c['nh4']['longest_observed_run_mae_minutes']<=b['nh4']['longest_observed_run_mae_minutes']*1.05}
            for name in ('nh4','do'):
                check[name+'_recall_not_worse']=c[name]['step_recall']>=b[name]['step_recall']
                check[name+'_precision_within_2pp']=c[name]['step_precision']>=b[name]['step_precision']-.02
                check[name+'_fpr_within_1pp']=c[name]['fpr']<=b[name]['fpr']+.01
                check[name+'_mae_within_5pct']=c[name]['mae']<=b[name]['mae']*1.05
                check[name+'_one_step_mae_within_5pct']=c1[name]['mae']<=b1[name]['mae']*1.05
            checks.append(check)
    gains={base:float(np.mean([next(r for r in rows if r['mode']==candidate and r['seed']==seed and r['split']==split and r['reset_steps']==275)['nh4']['step_recall']-
          next(r for r in rows if r['mode']==base and r['seed']==seed and r['split']==split and r['reset_steps']==275)['nh4']['step_recall'] for seed in CFG['seeds']])) for base in ('reference','single_mse')}
    passed=all(v for c in checks for k,v in c.items() if k not in ('seed','base')) and min(gains.values())>=.05
    absolute=all(r[name]['step_recall']>=.8 and r[name]['fpr']<=.1 for r in rows
                 if r['mode']==candidate and r['split']==split and r['reset_steps']==275 for name in ('nh4','do'))
    return {'candidate':candidate,'split':split,'per_seed':checks,'mean_nh4_recall_gain':gains,
            'screen_passed':bool(passed),'absolute_recall80_fpr10':bool(absolute),
            'pilot_eligibility':bool(passed and absolute)}


def evaluate(model,xs,aux,mode,seed):
    rows=[]
    for split,x in xs.items():
        for reset in (1,36,275):
            pred=replay(model,x,aux[split],'with_routing',reset)
            rows.append({'mode':mode,'seed':seed,'split':split,'reset_steps':reset,**score(pred,x)})
    return rows


def preflight():
    pred=torch.tensor([[3.,1.],[5.,3.]],requires_grad=True)
    target=torch.tensor([[5.,3.],[3.,1.]])
    loss=balanced_boundary(pred,target,torch.tensor([.1,.05]),torch.tensor([.2,.1]))
    loss.backward()
    assert (pred.grad[0]<0).all() and (pred.grad[1]>0).all() and torch.isfinite(pred.grad).all()
    print('PASS boundary loss corrects both missed violations and false alarms',flush=True)


def main(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    preflight()
    if a.preflight_only:return
    if a.out.exists() and any(a.out.iterdir()):raise ValueError(f'Refusing overwrite {a.out}')
    a.out.mkdir(parents=True)
    data=pd.read_pickle(a.data);days=data['days']['train'];assert len(days)==330
    partition={'fit':days[:263],'purge':days[263:264],'inner_holdout':days[264:]}
    xs={s:arrays({'data':data['data'],'days':partition},s) for s in ('fit','inner_holdout')}
    frame=routing_frame(a.raw)
    aux={s:np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in partition[s]]) for s in xs}
    for split in xs:
        for d in partition[split]:
            index=data['data'].loc[d].index
            assert len(index)==288 and np.all(np.diff(index.asi8)==300*10**9)
    state,exog=pack(xs['fit'])
    tensors=tuple(torch.tensor(z).float() for z in (state,exog,aux['fit']))
    scale=tensors[0].reshape(-1,2).std(0,unbiased=False)+1e-6
    prevalence=(tensors[0][:,13:]>torch.tensor(CFG['thresholds'])).float().mean((0,1))
    assert ((prevalence>0)&(prevalence<1)).all()
    report={'started_at':datetime.now(timezone.utc).isoformat(),'config':CFG,'partition':partition,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.raw,a.data)},
        'state_scale':scale.tolist(),'prevalence':prevalence.tolist(),'test_used':False,
        'outer_val_used':False,'internal_holdout_reused':True,'training':[],'calibration':[],'evaluation':[]}
    calibration,_=window(tensors,np.random.default_rng(CFG['calibration_seed']))
    for seed in CFG['seeds']:
        source=a.pretrained/f'with_routing_s{seed}.pt'
        original=load_model(source)
        report['evaluation']+=evaluate(original,xs,aux,'reference',seed)
        mse,boundary=losses(original,*calibration,scale,prevalence,False)
        gm=torch.autograd.grad(mse,tuple(original.parameters()),retain_graph=True)
        gb=torch.autograd.grad(boundary,tuple(original.parameters()))
        norm=lambda g:torch.sqrt(sum(t.square().sum() for t in g)).item()
        alpha=CFG['boundary_gradient_ratio']*norm(gm)/max(norm(gb),1e-12)
        report['calibration'].append({'seed':seed,'alpha':alpha,'mse_grad_norm':norm(gm),'boundary_grad_norm':norm(gb),
            'pretrained_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        for mode in MODES:
            model=load_model(source)
            opt=torch.optim.Adam(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
            rng=np.random.default_rng(CFG['sampling_seed_base']+seed);digest=hashlib.sha256();logs=[]
            for update in tqdm(range(CFG['updates']),desc=f'{mode} seed {seed}',mininterval=5):
                batch,signature=window(tensors,rng);digest.update(signature)
                mse,boundary=losses(model,*batch,scale,prevalence,mode.startswith('multi'))
                loss=mse+(alpha*boundary if mode.endswith('boundary') else 0.)
                opt.zero_grad();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['grad_clip'],error_if_nonfinite=True)
                assert torch.isfinite(loss)
                opt.step();logs.append({'mse':mse.item(),'boundary':boundary.item(),'grad':grad.item()})
            path=a.out/f'{mode}_s{seed}.pt';torch.save(model.state_dict(),path)
            restored=load_model(path)
            torch.testing.assert_close(model(*[v[:,0] if i!=2 else v[:,0].reshape(-1,4,2) for i,v in enumerate(calibration)]),
                restored(*[v[:,0] if i!=2 else v[:,0].reshape(-1,4,2) for i,v in enumerate(calibration)]),atol=0,rtol=0)
            report['training'].append({'mode':mode,'seed':seed,'batch_sha256':digest.hexdigest(),
                'initial_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'first':logs[0],'last':logs[-1]})
            write(a.out/f'{mode}_s{seed}_log.json',logs)
            report['evaluation']+=evaluate(restored,xs,aux,mode,seed)
            write(a.out/'progress.json',report)
    for seed in CFG['seeds']:
        rows=[r for r in report['training'] if r['seed']==seed]
        assert len({r['batch_sha256'] for r in rows})==len({r['initial_sha256'] for r in rows})==1
    report['gates']=[gates(report['evaluation'],'inner_holdout',mode) for mode in MODES if mode!='single_mse']
    report['checks']={'same_initial_weights':True,'same_windows':True,'finite_training':True,'restore_exact':True}
    report['completed_at']=datetime.now(timezone.utc).isoformat();write(a.out/'results.json',report)
    print(json.dumps([{k:v for k,v in g.items() if k!='per_seed'} for g in report['gates']]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','data','pretrained','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    main(p.parse_args())
