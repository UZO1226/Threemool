"""Independent archive replay and summary of e84 perfect-information choices."""
import argparse
import json
from pathlib import Path

import numpy as np

from choice import MODES, ROOT, TOL


def summarize(dest,root):
    data=json.loads((dest/'summary.json').read_text())
    records=data['records']
    labels={'episode':'구성원별 선택, 겨울 평균 제약','date':'날짜별 선택, 겨울 평균 제약','date_strict':'날짜별 선택, 매 날짜 제약'}
    def desc(f):
        return '최초 증량 유지' if f=='after_pulse' else '최초 증량 생략 허용'
    lines=['# e84 기존 경로에서 얻을 수 있는 최대 공기 절감','',
           '공기는 m³/23.75h이며 양의 절감량은 기본 FCSRL보다 적게 사용했다는 뜻이다. 각 seed의 기본 FCSRL이 세 수질 제약의 기준이다. 최선 경로는 사후에 선택했다.','',
           '| 선택지 | 정보와 제약 | 세 수질 제약 가능 | 공동 개선 | 5 seed 평균 절감 | seed별 절감률 평균 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for fam in ('after_pulse','skip_allowed'):
        for scope in labels:
            rr=[r for r in records if r['family']==fam and r['scope']==scope and r['status']=='optimal']
            wins=sum(r['delta'][0]<-TOL for r in rr)
            avg=f"{-np.mean([r['delta'][0] for r in rr]):.2f}" if len(rr)==5 else '산출 불가'
            pct=f"{np.mean([r['saving_pct'] for r in rr]):.3f}%" if len(rr)==5 else '산출 불가'
            lines.append(f'| {desc(fam)} | {labels[scope]} | {len(rr)}/5 | {wins}/5 | {avg} | {pct} |')
    lines+=['','## seed별 집계','',
            '수질 열은 선택 경로에서 기본 경로를 뺀 값이다. 악화 일수는 그 날짜에서 5개 구성원 평균 수질 중 하나라도 기본보다 높았던 일수다. 불가능한 날짜는 날짜별 제약에서 세 지표를 지키는 선택지가 없는 날이다.','',
            '| 선택지 | 조건 | seed | 절감 | NH4 위반 차 | DO 위반 차 | NH4 초과 차 | 악화 또는 불가능 일수 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    extra=[]
    for seed in range(5):
        archive=[]
        for m in MODES:
            exp='e83' if m=='limited' else 'e81'
            with np.load(root/exp/'main'/f'{seed}_{m}.npz') as z:
                np.testing.assert_array_equal(z['dates'],data['dates'])
                state=z['states']
                archive.append(np.stack([z['air'].mean(1)*23.75,
                    np.count_nonzero(state[:,:,0]>4,axis=1),
                    np.any(state[:,:,2:]>2,axis=-1).sum(1),
                    np.clip(state[:,:,0]-4,0,None).mean(1)*23.75],axis=1).reshape(40,5,4))
        v=np.stack(archive,axis=2)
        delta=v-v[:,:,:1]
        np.testing.assert_allclose(v.mean((0,1)),data['fixed'][seed],rtol=0,atol=1e-8)
        for r in [r for r in records if r['seed']==seed]:
            tag=f"| {desc(r['family'])} | {labels[r['scope']]} | {seed} | "
            if r['status']=='optimal':
                ids=np.array(r['selected'])
                if r['scope']!='episode':
                    ids=np.repeat(ids,5)
                picked=delta.reshape(200,4,4)[np.arange(200),ids].reshape(40,5,4).mean(1)
                dif=picked.mean(0)
                np.testing.assert_allclose(dif,r['delta'],atol=1e-8,rtol=0)
                days=int((picked[:,1:]>TOL).any(1).sum())
                assert days==r['quality_worse_days']
                lines.append(tag+f'{-dif[0]:.2f} | {dif[1]:+.4f} | {dif[2]:+.4f} | {dif[3]:+.4f} | {days} |')
            else:
                failed=r.get('failed_units')
                if failed is not None:
                    ok=(delta[:,:,1:,1:].mean(1)<=TOL).all(2).any(1)
                    np.testing.assert_array_equal(np.flatnonzero(~ok),failed)
                lines.append(tag+'불가능 | | | | '+(str(len(failed)) if failed is not None else '')+' |')
        # Report a transparent impossibility certificate when one metric alone rules out a solution.
        minima=delta[:,:,1:,1:].min(2).mean((0,1))
        day_pulse=delta[:,:,1].mean(1)
        useful=(day_pulse[:,1:]<=TOL).all(1)&(day_pulse[:,0]<-TOL)
        pair_save=-np.minimum(np.where(useful,day_pulse[:,0],0),0).mean()
        extra.append([seed,*minima,int(useful.sum()),pair_save])
    lines+=['','## 사후 해석 검산','',
            '앞 세 열은 최초 증량을 유지할 때 각 수질 지표만 단독으로 최소화한 겨울 평균 차이다. 한 열이라도 양수이면 그 지표 하나만으로 불가능함을 확인할 수 있다. 마지막 두 열은 기본과 단발 증량만 허용하고 날짜별 수질을 지킨 경우다. 해당 날짜는 5개 모델 평균 기준이다.','',
            '| seed | 최소 NH4 위반 차 | 최소 DO 위반 차 | 최소 NH4 초과 차 | 증량으로 공동 개선되는 날 / 40 | 전체 날짜 평균 절감 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s,a,b,c,n,sv in extra:
        lines.append(f'| {s} | {a:+.4f} | {b:+.4f} | {c:+.4f} | {n} | {sv:.2f} |')
    lines.append(f'| 평균 | | | | | {np.mean([r[-1] for r in extra]):.2f} |')
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args()
    summarize(args.dest,args.root)
