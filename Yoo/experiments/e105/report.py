"""Compare hour-three levels and two-hour increments on the same dated events."""
import argparse
import json
from pathlib import Path

import numpy as np

LABELS = ('미복원','NH4만','DO만','NH4와 DO','전 상태')
MODES = ('none','nh4','do','nh4_do','all')


def export(root,dest):
    draws = ((np.random.default_rng(104).integers(40,size=(2000,6))[:,:,None]
              +np.arange(7))%40).reshape(2000,-1)[:,:40]
    results = {}
    lines = ['# e105 1시간 실측 복원과 3시간 예측', '',
             'NH4 단위는 mg N/L, 최대 DO MAE 단위는 mg O₂/L다. 모형별 예측을 먼저 평균하고 '
             '각 날짜 안에서 창 평균을 낸 뒤 날짜에 같은 비중을 줬다. 각 조건에서 1~3시간 변화는 '
             '복원 직후의 해당 모형 NH4를 기준으로 삼았다.', '']
    with np.load(root/'reset.npz') as z:
        assert tuple(z['modes'])==MODES
        for label, mask in [('증가',z['dq']>0),('감소',z['dq']<0)]:
            days=z['day'][mask]
            available, inv, counts=np.unique(days,return_inverse=True,return_counts=True)
            def daily(x):
                arr=np.full((40,*x.shape[1:]),np.nan)
                arr[available]=np.stack([x[days==d].mean(0) for d in available])
                return arr
            def avg(x):
                v=np.nanmean(daily(x),axis=0)
                np.testing.assert_allclose(v,np.average(x,axis=0,weights=1/counts[inv]),atol=1e-10)
                return v
            def contrast(x):
                samples=daily(x)[draws]
                good=np.isfinite(samples).any(1)
                lo,hi=np.quantile(np.nanmean(samples[good],axis=1),[.025,.975])
                return [float(avg(x)),float(lo),float(hi)]
            state=z['truth'][mask]
            y0,y1,y3=state[:,:,0].T
            end=z['ends'][mask]
            starts=z['starts'][mask,:,:,0].mean(2)
            pred=end[:,:,:,0].mean(2)
            level_err=abs(pred-y3[:,None])
            inc=pred-starts
            inc_err=abs(inc-(y3-y1)[:,None])
            mae=avg(level_err)
            change=avg(pred-y0[:,None])
            segment=avg(inc)
            segment_mae=avg(inc_err)
            do=avg(abs(end[:,:,:,2:].max(-1).mean(2)-state[:,2,2:].max(-1)[:,None]))
            member=avg(abs(end[:,:,:,0]-y3[:,None,None]))
            summary = dict(events=len(days),days=len(available),observed_change=float(avg(y3-y0)),
                           observed_segment=float(avg(y3-y1)),hold_1h_mae=float(avg(abs(y3-y1))),modes={})
            lines += [f"## 공기 {label}, {len(days)}개 창 및 {len(available)}일", '',
                      f"실측 NH4 변화는 처음부터 3시간까지 {summary['observed_change']:+.3f}, 1~3시간에는 {summary['observed_segment']:+.3f}였다. "
                      f"1시간 실측 NH4 유지 기준의 최종 MAE는 {summary['hold_1h_mae']:.3f}였다.", '',
                      '| 교체 항목 | 최종 NH4 MAE | 0~3시간 예측 변화 | 1~3시간 예측 변화 | 1~3시간 변화 MAE | 최종 최대 DO MAE |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |']
            for j,name in enumerate(MODES):
                summary['modes'][name]=dict(mae=mae[j],change=change[j],segment=segment[j],segment_mae=segment_mae[j],
                                            do_mae=do[j],member_mae=member[j].tolist())
                lines.append(f'| {LABELS[j]} | {mae[j]:.3f} | {change[j]:+.3f} | {segment[j]:+.3f} | {segment_mae[j]:.3f} | {do[j]:.3f} |')
            lines += ['', '앞 조건 MAE에서 뒤 조건 MAE를 뺀 차이와 탐색 95% 구간이다. 양수면 뒤 조건이 더 정확했다.', '',
                      '| 앞 조건 → 뒤 조건 | 최종 NH4 오차 감소 | 1~3시간 변화 오차 감소 |',
                      '| --- | ---: | ---: |']
            summary['contrasts']={}
            for a,b in [(0,1),(0,2),(2,1),(0,3),(0,4),(3,4)]:
                diffs=[contrast(err[:,a]-err[:,b]) for err in (level_err,inc_err)]
                summary['contrasts'][f'{MODES[a]}_minus_{MODES[b]}']=dict(zip(('level','segment'),diffs))
                cols=[f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]' for m,lo,hi in diffs]
                lines.append(f'| {LABELS[a]} → {LABELS[b]} | '+' | '.join(cols)+' |')
            lines += ['', '| 교체 항목 | 최종 NH4 MAE, 모델 0 / 1 / 2 / 3 / 4 |','| --- | --- |']
            for j,name in enumerate(LABELS):
                lines.append(f'| {name} | '+' / '.join(f'{v:.3f}' for v in member[j])+' |')
            lines.append('')
            results[label]=summary
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(results,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    (dest/'results.md').write_text('\n'.join(lines).rstrip()+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('dest',type=Path)
    args=p.parse_args()
    export(args.root,args.dest)
