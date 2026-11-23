"""Audit weighted ranks and render the chronological diagnostic."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def fmt(value, scale=1):
    return '미정' if value is None else f'{value*scale:.3f}'


def report(root,dest):
    data=json.loads((root/'summary.json').read_text())
    lines=['# e85 시간순 확인 결과','','뒤 48일의 e78 감량 전용 궤적에서 실제 감량이 제안된 시점을 평가했다. '
           '악화는 같은 시작점의 원 명령보다 6시간 NH4 위반이 1스텝 이상 또는 NH4 초과 적분이 0.1 mg h/L 이상 늘어난 경우다. '
           'AUC와 악화율은 같은 날짜의 전체 가중치를 1로 두었다. 각 군의 악화율은 이 가중치로 조건부 평균했다.','',
           '| seed | 앞 140일 보정 건수 | 뒤 48일 적격 건수 / 날짜 | 악화 건수 | 차단 건수 | 허용군 악화율 % | 차단군 악화율 % | 위험 순위 AUC | 통과 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in data['rows']:
        with np.load(root/f"probe{r['seed']}.npz") as z:
            ids=z['keys']
            assert len(np.unique(ids,axis=0))==r['n'] and len(set(z['dates']))==48
            assert len(set(ids[:,0]))==r['dates'] and np.all(ids[:,0]<48)
            assert np.all(z['fit_keys'][:,0]<140) and np.median(z['fit_scores'])==r['threshold']
            gap=z['cut']-z['normal']
            change=z['totals'][:,1,2]-z['totals'][:,0,2]
            harmed=(change[:,1]>=1)|(change[:,3]>=.1)
            counts=Counter(ids[:,0])
            weight=np.array([1/counts[d] for d in ids[:,0]])
            reject=gap>r['threshold']
            assert int(harmed.sum())==r['positive'] and int(reject.sum())==r['blocked']
            if harmed.any() and not harmed.all():
                # Direct pairwise concordance, including half credit for ties.
                pair_weight=weight[harmed,None]*weight[~harmed]
                diff=gap[harmed,None]-gap[~harmed]
                auc=np.sum(pair_weight*((diff>0)+.5*(diff==0)))/pair_weight.sum()
                np.testing.assert_allclose(auc,r['auc'],atol=1e-12)
            for mask,key in [(reject,'blocked_rate'),(~reject,'allowed_rate')]:
                if mask.any():
                    np.testing.assert_allclose(weight[mask&harmed].sum()/weight[mask].sum(),r[key],atol=1e-12)
                else:
                    assert r[key] is None
        lines.append(f"| {r['seed']} | {r['fit_n']} | {r['n']} / {r['dates']} | {r['positive']} | {r['blocked']} | "+
                     f"{fmt(r['allowed_rate'],100)} | {fmt(r['blocked_rate'],100)} | {fmt(r['auc'])} | {'통과' if r['passed'] else '미달'} |")
    n=sum(r['passed'] for r in data['rows'])
    assert (n>=4)==data['advance']
    lines+=['',f'통과 {n}/5. 사전 조건은 4/5 이상이다.','',
            '| seed | 고정 문턱 | 악화가 있던 날짜 | 비악화가 있던 날짜 | 감량의 6시간 공기 차 m³ | NH4 위반 차 | DO 위반 차 | NH4 초과 차 mg h/L |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in data['rows']:
        lines.append(f"| {r['seed']} | {r['threshold']:.7f} | {r['class_days'][1]} | {r['class_days'][0]} | "+
                     ' | '.join(f'{v:+.4f}' for v in r['delta'])+' |')
    lines+=['','마지막 표의 차이는 전체 적격 시작점에서 단발 감량 빼기 원 명령의 날짜가중 평균이다. 이후에는 두 분기 모두 같은 기본 FCSRL로 운전했다. 같은 날짜에 악화 및 비악화 사례가 함께 있을 수 있다.']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args()
    report(a.root,a.dest)
