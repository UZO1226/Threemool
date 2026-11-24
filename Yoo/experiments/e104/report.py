"""Score frozen forecasts against the same measured endpoint, weighting dates equally."""
import argparse
import json
from pathlib import Path

import numpy as np

STEPS = (1,4,12)
NAMES = ('NH4','DOmax','DO1','DO2','DO3')


def signals(state):
    return np.stack((state[...,0],state[...,2:].max(-1),*np.moveaxis(state[...,2:],-1,0)),axis=-1)


def scores(truth,rec,local,day,draws):
    dates = np.unique(day)
    def daily(x):
        return np.stack([x[day==d].mean(0) for d in dates])
    def avg(x):
        return daily(x).mean(0)
    def interval(x):
        per = np.full((40,*x.shape[1:]),np.nan)
        per[dates] = daily(x)
        samples = per[draws]
        good = np.isfinite(samples[:,:,0]).any(1)
        return np.quantile(np.nanmean(samples[good],axis=1),[.025,.975],axis=0).T
    records = []
    for j,h in enumerate(STEPS):
        y = truth[:,h]
        r,l = rec[:,j],local[:,j]
        rbar,lbar = r.mean(1),l.mean(1)
        er,el = abs(rbar-y),abs(lbar-y)
        ep,eq = abs(truth[:,0]-y),abs(truth[:,h-1]-y)
        stats = dict(rec=avg(er),local=avg(el),start_hold=avg(ep),last_hold=avg(eq),
                     obs_change=avg(y-truth[:,0]),rec_change=avg(rbar-truth[:,0]),
                     outside=avg((y<r.min(1)-1e-8)|(y>r.max(1)+1e-8)),
                     member_rec=avg(abs(r-y[:,None])).T,member_local=avg(abs(l-y[:,None])).T,
                     rec_minus_hold_ci=interval(er-ep),rec_minus_local_ci=interval(er-el),
                     local_minus_last_ci=interval(el-eq))
        for s,name in enumerate(NAMES):
            records.append(dict(signal=name,minutes=h*15,**{k:v[s].tolist() for k,v in stats.items()}))
    return records


def export(root,dest):
    rng=np.random.default_rng(104)
    draws=((rng.integers(40,size=(2000,6))[:,:,None]+np.arange(7))%40).reshape(2000,-1)[:,:40]
    result = {}
    lines = ['# e104 실제 폭기 변화 뒤 수질 예측 오차', '',
             'MAE는 날짜 가중 평균 절대오차(mg/L)다. NH4는 질소, DO는 산소 농도다. '
             '5개 예측의 평균을 평가하며 DOmax는 각 구성원에서 먼저 최댓값을 구한다. '
             '반복은 최초 실측 상태에서 출발한다. 복원은 목표 직전까지의 실측 상태를 안다. '
             '95% 구간은 달력 순서 검증일 7개씩을 묶은 순환 블록 복원추출의 탐색 구간이다.', '']
    with np.load(root/'replay.npz') as z:
        truth,rec,local = [signals(z[k]) for k in ('truth','rec','local')]
        for label,mask in [('증가',z['dq']>0),('감소',z['dq']<0)]:
            day=z['date_index'][mask]
            rs=scores(truth[mask],rec[mask],local[mask],day,draws)
            result[label]=dict(events=int(mask.sum()),days=len(np.unique(day)),metrics=rs)
            lines += [f"## 공기 {label} ({mask.sum()}개 변화, {len(np.unique(day))}일)", '',
                      '| 측정값 | 분 | 반복 MAE | 최초 농도 유지 | 복원 MAE | 직전 농도 유지 | 반복 빼기 복원 [95% 구간] | 반복 빼기 최초 유지 [95% 구간] |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for v in rs:
                def bounds(value,key):
                    lo,hi=v[key]
                    return f'{value:+.3f} [{lo:+.3f}, {hi:+.3f}]'
                gap=bounds(v['rec']-v['local'],'rec_minus_local_ci')
                gain=bounds(v['rec']-v['start_hold'],'rec_minus_hold_ci')
                lines.append(f"| {v['signal']} | {v['minutes']} | {v['rec']:.3f} | {v['start_hold']:.3f} | {v['local']:.3f} | {v['last_hold']:.3f} | {gap} | {gain} |")
            lines += ['', '| 측정값 | 분 | 실측 변화 | 평균 예측 변화 | 실측이 5개 예측 범위 밖 | 반복 MAE, 모델 0 / 1 / 2 / 3 / 4 |',
                      '| --- | ---: | ---: | ---: | ---: | --- |']
            for v in rs:
                ms=' / '.join(f'{a:.3f}' for a in v['member_rec'])
                lines.append(f"| {v['signal']} | {v['minutes']} | {v['obs_change']:+.3f} | {v['rec_change']:+.3f} | {v['outside']:.1%} | {ms} |")
            lines.append('')
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'results.md').write_text('\n'.join(lines).rstrip()+'\n')
    (dest/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('dest',type=Path)
    args=p.parse_args()
    export(args.root,args.dest)
