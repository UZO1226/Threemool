"""Check saved episodes and present the two prespecified paired comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np


def numbers(v):
    return ' | '.join(f'{n:+.4f}' for n in v)


def report(root, controls, dest):
    records=json.loads((root/'records.json').read_text())
    names=['base','recover','limited']
    keys=['air','nh','do','excess','limit_pct']
    table={}
    lines=['# e83 겨울 하루 운전 결과','','40일, GBM 구성원 5개, 정책 seed 5개. 각 seed의 200개 에피소드 전체를 평균했다. base는 기본 FCSRL, recover는 e81, limited는 상승 제한 추가다.','',
           '| 방식 | 공기 m³/23.75h | NH4 위반 스텝 | DO 위반 스텝 | NH4 초과 mg h/L | 합산 한도 12 충족 % |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in names:
        values=[]
        for seed in range(5):
            r=next(r for r in records if r['seed']==seed and r['mode']==m)
            path=(root if m=='limited' else controls)/f'{seed}_{m}.npz'
            with np.load(path) as z:
                assert z['states'].shape==(200,95,5) and len(set(z['dates']))==40
                s=z['states']
                nh=np.count_nonzero(s[:,:,0]>4,axis=1)
                do=(s[:,:,2:]>2).any(2).sum(1)
                a=[z['air'].sum()/800,nh.mean(),do.mean(),np.clip(s[:,:,0]-4,0,None).sum()/800,
                   100*np.mean(nh+do<=12)]
                np.testing.assert_allclose(a,[r[k] for k in keys],atol=1e-8,rtol=0)
                if m=='limited':
                    np.testing.assert_allclose(z['withheld'].sum()/800,r['withheld_volume'],atol=1e-8)
                    assert np.count_nonzero(z['withheld'])==r['clipped_steps']
                values.append(a)
        table[m]=np.array(values)
        lines.append('| '+m+' | '+' | '.join(f'{x:.4f}' for x in table[m].mean(0))+' |')
    for ref in ['recover','base']:
        diff=table['limited']-table[ref]
        rows=np.vstack([diff,diff.mean(0)])
        wins=(rows[:,0]<-1e-8)&(rows[:,1:4]<=1e-8).all(1)
        lines+=['',f'## limited 빼기 {ref}','','| seed | 공기 차 | NH4 위반 차 | DO 위반 차 | NH4 초과 차 | 한도 충족률 차 %p | 공동 개선 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for i,d in enumerate(rows):
            mark='충족' if wins[i] else '미달'
            if i==5:
                mark+=f', {sum(wins[:5])}/5 seed'
            lines.append(f"| {i if i<5 else '평균'} | "+numbers(d)+f' | {mark} |')
    lines+=['','## 제한이 실제 작동한 범위','',
            '| seed | 제한 에피소드 / 200 | 제한 스텝 | NH4 예외로 통과한 상승 스텝 | 평균 직접 억제 부피 m³/23.75h |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        if r['mode']=='limited':
            lines.append(f"| {r['seed']} | {r['clipped_episodes']} | {r['clipped_steps']} | {r['bypassed_steps']} | {r['withheld_volume']:.3f} |")
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args()
    report(a.root,a.controls,a.dest)
