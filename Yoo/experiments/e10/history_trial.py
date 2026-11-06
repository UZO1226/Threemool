"""과거 상태 입력의 효과를 학습 구간 내부의 시간 분할로 대조한다."""
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
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e09'))
from model_trial import pack
from audit import arrays, simulate, Surrogate, write

LAGS = (1, 3, 6, 12)
CONFIG = dict(epochs=60, batch_size=1024, lr=.001, weight_decay=.00001,
              grad_clip=1., seeds=[0, 1], lags=list(LAGS), hidden=[64, 64],
              checkpoint='last', fit_days=263, purge_days=1, holdout_days=66)


def features(state, exog, history, do_next=None):
    n, d = state.unbind(-1)
    air, phase, temp, flow, s, c = exog.unbind(-1)
    ph = torch.stack([(phase == k).to(state.dtype) for k in (0, 1, 2)], -1)
    common = torch.stack((temp, flow / 1000, s, c), -1)
    if do_next is None:
        head = torch.stack((d, air / 1000), -1)
        base = torch.cat((head, ph, temp[:, None], n[:, None], common[:, 1:]), -1)
    else:
        base = torch.cat((torch.stack((n, d, do_next), -1), ph, common), -1)
    return torch.cat((base, history.flatten(1)), -1)


def batches(x, mode):
    state, exog = pack(x)
    state, exog = torch.tensor(state).float(), torch.tensor(exog).float()
    current, target = state[:, 12:-1], state[:, 13:]
    history = torch.stack([state[:, 12-lag:288-1-lag] for lag in LAGS], -2)
    if mode == 'current_repeat':
        history = current[:, :, None].expand(-1, -1, 4, -1)
    return (current.reshape(-1, 2), exog[:, 12:-1].reshape(-1, 6),
            history.reshape(-1, 4, 2), target.reshape(-1, 2))


class Model(nn.Module):
    def __init__(self, batch):
        super().__init__()
        state, exog, history, target = batch
        xd, xn = features(state, exog, history), features(state, exog, history, target[:, 1])
        for name, value in [('do_mu', xd.mean(0)), ('do_sd', xd.std(0, unbiased=False) + 1e-6),
                            ('nh4_mu', xn.mean(0)), ('nh4_sd', xn.std(0, unbiased=False) + 1e-6),
                            ('delta_mu', (target-state).mean(0)),
                            ('delta_sd', (target-state).std(0, unbiased=False) + 1e-6)]:
            self.register_buffer(name, value)
        self.do_net = nn.Sequential(nn.Linear(18,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.nh4_net = nn.Sequential(nn.Linear(18,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self, state, exog, history):
        xd = features(state, exog, history)
        dn = (state[:, 1] + self.do_net((xd-self.do_mu)/self.do_sd)[:,0]
              * self.delta_sd[1] + self.delta_mu[1]).clamp_min(0)
        xn = features(state, exog, history, dn)
        nn_ = (state[:, 0] + self.nh4_net((xn-self.nh4_mu)/self.nh4_sd)[:,0]
               * self.delta_sd[0] + self.delta_mu[0]).clamp_min(0)
        return torch.stack((nn_,dn),-1)


@torch.no_grad()
def rollout(model, x, mode, reset):
    state, exog = pack(x)
    state, exog = torch.tensor(state).float(), torch.tensor(exog).float()
    buffer = state[:, :13].clone()
    result=[]
    for t in range(12,287):
        if (t-12) % reset == 0:
            buffer = state[:, t-12:t+1].clone()
        history = torch.stack([buffer[:, -1-lag] for lag in LAGS], 1)
        if mode == 'current_repeat':
            history = buffer[:, -1, None].expand(-1,4,-1)
        pred = model(buffer[:,-1],exog[:,t],history)
        assert torch.isfinite(pred).all()
        result.append(pred.numpy())
        buffer = torch.cat((buffer[:,1:],pred[:,None]),1)
    return np.stack(result,1)


def runs(x):
    change = np.diff(np.r_[False, x, False].astype(int))
    return list(zip(np.flatnonzero(change==1).tolist(),np.flatnonzero(change==-1).tolist()))


def event_metrics(pred, truth, initial):
    out={}
    for k, name, limit in ((0,'nh4',4),(1,'do',2)):
        p, y = pred[:,:,k]>limit, truth[:,:,k]>limit
        tp, fp, fn = int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum())
        precision, recall = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.
        actual_events=pred_events=matched=0
        timing, duration, longest=[],[],[]
        for i in range(len(y)):
            # 평가 시작 전 관측값을 붙여 시작 시점의 검열된 사건을 구분한다.
            yr=runs(np.r_[initial[i,k]>limit,y[i]])
            pr=runs(np.r_[initial[i,k]>limit,p[i]])
            longest.append(abs(max([b-max(a,1) for a,b in yr] or [0])-
                               max([b-max(a,1) for a,b in pr] or [0]))*5)
            yr=[(a,b) for a,b in yr if a>0]
            pr=[(a,b) for a,b in pr if a>0]
            actual_events+=len(yr);pred_events+=len(pr)
            candidates=sorted((abs(a-c),j,l) for j,(a,b) in enumerate(yr)
                               for l,(c,d) in enumerate(pr) if abs(a-c)<=3)
            used_y,used_p=set(),set()
            for distance,j,l in candidates:
                if j in used_y or l in used_p:continue
                used_y.add(j);used_p.add(l);matched+=1
                a,b=yr[j];c,d=pr[l]
                timing.append(abs(a-c)*5)
                if b<len(y[i])+1 and d<len(y[i])+1:
                    duration.append(abs((b-a)-(d-c))*5)
        out[name]={'mae':float(abs(pred[:,:,k]-truth[:,:,k]).mean()),
                   'predicted_steps':float(p.sum(1).mean()), 'actual_steps':float(y.sum(1).mean()),
                   'daily_steps_mae':float(abs(p.sum(1)-y.sum(1)).mean()),
                   'step_precision':precision,'step_recall':recall,
                   'step_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                   'actual_onsets':actual_events,'predicted_onsets':pred_events,'matched_onsets':matched,
                   'onset_f1':2*matched/(actual_events+pred_events) if actual_events+pred_events else 0.,
                   'matched_onset_mae_minutes':float(np.mean(timing)) if timing else None,
                   'uncensored_matched_duration_count':len(duration),
                   'uncensored_matched_duration_mae_minutes':float(np.mean(duration)) if duration else None,
                   'longest_observed_run_mae_minutes':float(np.mean(longest))}
    pc=(pred[:,:,0]>4).astype(int)+(pred[:,:,1]>2)
    yc=(truth[:,:,0]>4).astype(int)+(truth[:,:,1]>2)
    out['cost']={'predicted':float(pc.sum(1).mean()),'actual':float(yc.sum(1).mean()),
                 'daily_mae':float(abs(pc.sum(1)-yc.sum(1)).mean())}
    return out


def preflight():
    assert runs([False,True,True,False,True])==[(1,3),(4,5)]
    # 일치한 사건, 한 스텝 늦게 시작하고 같은 시점에 끝나는 사건, 누락 사건을 검증한다.
    y=np.zeros((1,8,2));y[0,2:5,0]=5
    assert event_metrics(y,y,np.zeros((1,2)))['nh4']['onset_f1']==1.
    p=y.copy();p[0,2,0]=0
    m=event_metrics(p,y,np.zeros((1,2)))['nh4']
    assert m['matched_onset_mae_minutes']==5 and m['uncensored_matched_duration_mae_minutes']==5
    assert event_metrics(np.zeros_like(y),y,np.zeros((1,2)))['nh4']['step_recall']==0.
    x={k:np.tile(np.arange(288,dtype=float),(2,1)) for k in ('nh4','do','air','phase','temp','q_in')}
    x['phase'][:]=2
    b=batches(x,'history');np.testing.assert_array_equal(b[2][0,:,0], [11,9,6,0])
    assert b[0][0,0]==12 and b[3][0,0]==13 and len(b[0])==550
    torch.manual_seed(0);model=Model(b)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.delta_mu.zero_()
    before=rollout(model,x,'history',275)
    changed={k:v.copy() for k,v in x.items()}
    changed['nh4'][:,13:]+=100
    changed['do'][:,13:]+=100
    np.testing.assert_array_equal(before,rollout(model,changed,'history',275))
    assert np.all(before==12)
    print('PASS event metrics, causal lag indices, no future state access in free run',flush=True)


def main(a):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    preflight()
    if a.preflight_only:return
    if a.out.exists() and any(a.out.iterdir()):raise ValueError(f'Refusing overwrite {a.out}')
    a.out.mkdir(parents=True)
    data=pd.read_pickle(a.data)
    days=data['days']['train'];assert len(days)==330
    partition={'fit':days[:263],'purge':days[263:264],'inner_holdout':days[264:]}
    assert pd.Timestamp(partition['inner_holdout'][0])-pd.Timestamp(partition['fit'][-1])>pd.Timedelta(days=1)
    for day in days:
        idx=data['data'].loc[day].index
        assert len(idx)==288 and idx[0].hour==idx[0].minute==0
        assert np.all(np.diff(idx.asi8)==300*10**9)
    temp={'data':data['data'],'days':partition}
    xs={k:arrays(temp,k) for k in ('fit','inner_holdout')}
    report={'started_at':datetime.now(timezone.utc).isoformat(),'config':CONFIG,'partition':partition,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'test_used':False,'outer_val_used_for_new_model_trial':False,
            'versions':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},
            'training':[],'evaluation':[]}
    for seed in CONFIG['seeds']:
        for mode in ('current_repeat','history'):
            torch.manual_seed(seed)
            b=batches(xs['fit'],mode);model=Model(b)
            initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
            opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
            rng=np.random.default_rng(10000+seed);digest=hashlib.sha256();logs=[]
            for epoch in tqdm(range(CONFIG['epochs']),desc=f'{mode} seed {seed}',mininterval=5):
                perm=rng.permutation(len(b[0]));digest.update(perm.tobytes());losses=[]
                for begin in range(0,len(perm),CONFIG['batch_size']):
                    ix=perm[begin:begin+CONFIG['batch_size']]
                    state,exog,history,target=[v[ix] for v in b]
                    pred=model(state,exog,history)
                    loss=((pred-target)/model.delta_sd).square().mean()
                    opt.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['grad_clip'],error_if_nonfinite=True)
                    assert torch.isfinite(loss)
                    opt.step();losses.append(loss.item())
                logs.append(float(np.mean(losses)))
            path=a.out/f'{mode}_s{seed}.pt';torch.save(model.state_dict(),path)
            restored=Model(b);restored.load_state_dict(torch.load(path,weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model(*[v[:32] for v in b[:3]]),restored(*[v[:32] for v in b[:3]]),rtol=0,atol=0)
            report['training'].append({'mode':mode,'seed':seed,'parameter_count':sum(p.numel() for p in model.parameters()),
                'initial_parameter_sha256':initial_hash,'batch_sha256':digest.hexdigest(),'epoch_losses':logs,
                'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for split,x in xs.items():
                truth=np.stack((x['nh4'][:,13:],x['do'][:,13:]),-1)
                initial=np.stack((x['nh4'][:,12],x['do'][:,12]),-1)
                for reset in (1,12,275):
                    pred=rollout(restored,x,mode,reset)
                    np.savez_compressed(a.out/f'{mode}_s{seed}_{split}_h{reset}.npz',pred=pred,truth=truth,initial=initial)
                    report['evaluation'].append({'mode':mode,'seed':seed,'split':split,'reset_steps':reset,
                        'days':len(truth),'evaluated_steps_per_day':275,**event_metrics(pred,truth,initial)})
            write(a.out/'progress.json',report)
    for seed in CONFIG['seeds']:
        pair=[r for r in report['training'] if r['seed']==seed]
        for key in ('initial_parameter_sha256','batch_sha256','parameter_count'):assert pair[0][key]==pair[1][key]
    # 사전 기준은 두 seed의 장기 비용 MAE 개선, 평균 5% 개선과 단기 및 사건 지표 보존이다.
    checks=[]
    for seed in CONFIG['seeds']:
        def row(mode,h):return next(r for r in report['evaluation'] if r['seed']==seed and r['mode']==mode
                                   and r['split']=='inner_holdout' and r['reset_steps']==h)
        c,h=row('current_repeat',275),row('history',275)
        c1,h1=row('current_repeat',1),row('history',1)
        checks.append({'seed':seed,'cost_mae_improved':h['cost']['daily_mae']<c['cost']['daily_mae'],
                       'onset_f1_not_worse':h['nh4']['onset_f1']>=c['nh4']['onset_f1'],
                       'longest_run_mae_not_worse':h['nh4']['longest_observed_run_mae_minutes']<=c['nh4']['longest_observed_run_mae_minutes'],
                       'do_mae_within_5pct':h['do']['mae']<=c['do']['mae']*1.05,
                       'one_step_nh4_mae_within_5pct':h1['nh4']['mae']<=c1['nh4']['mae']*1.05})
    means={mode:np.mean([r['cost']['daily_mae'] for r in report['evaluation'] if r['split']=='inner_holdout'
                        and r['reset_steps']==275 and r['mode']==mode]) for mode in ('current_repeat','history')}
    report['gate']={'per_seed':checks,'mean_cost_mae_improvement_fraction':1-means['history']/means['current_repeat']}
    report['gate']['passed']=all(v for r in checks for k,v in r.items() if k!='seed') and report['gate']['mean_cost_mae_improvement_fraction']>=.05
    report['checks']={'paired_initial_parameters':True,'paired_batches':True,'finite_training':True,'save_restore_exact':True}
    report['completed_at']=datetime.now(timezone.utc).isoformat()
    write(a.out/'results.json',report)
    print(json.dumps(report['gate']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    main(p.parse_args())
