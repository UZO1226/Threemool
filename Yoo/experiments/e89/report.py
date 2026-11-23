"""Reconstruct the matches with row-wise distances, then summarize seed-wise diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(root, records, source):
    for r in records:
        s = r['seed']
        with np.load(root/f'matches{s}.npz') as data, np.load(source/f'probe{s}.npz') as old:
            tr_days = data['train_keys'][:,0]
            days = data['keys'][:,0]
            counts = np.bincount(tr_days)
            w = 1/counts[tr_days]
            effects = old['train_totals'][:,1]-old['train_totals'][:,0]
            labels = (effects[:,0]<=-25)&(effects[:,1:]<=1e-8).all(axis=1)
            for j,(name,width) in enumerate([('current',11),('history',14)]):
                library = data['train_x'][:,:width]
                scale = np.sqrt(np.average((library-np.average(library,weights=w,axis=0))**2,weights=w,axis=0))
                scale[scale==0]=1
                chosen = data['ids'][j]
                for i,x in enumerate(data['x'][:,:width]):
                    dist=np.sqrt(np.mean(((library-x)/scale)**2,axis=1))
                    rank=dist.argsort(kind='stable')
                    _,first=np.unique(tr_days[rank],return_index=True)
                    ids=rank[np.sort(first)[:5]]
                    np.testing.assert_array_equal(ids,chosen[i])
                    np.testing.assert_allclose(dist[ids],data['distances'][j,i])
                truth=old['positive']
                mismatch=labels[chosen[:,0]]!=truth
                loss=(labels[chosen].mean(axis=1)-truth)**2
                for group in ['all','supported','close']:
                    ix=data[group]
                    metrics=r['metrics'][group][name]
                    def equal_days(v):
                        return np.mean([v[ix & (days==d)].mean() for d in np.unique(days[ix])])
                    np.testing.assert_allclose([equal_days(mismatch),equal_days(loss)],[metrics['mismatch'],metrics['brier']])


def report(root, dest, source):
    records=json.loads((root/'summary.json').read_text())
    audit(root,records,source)
    lines=['# e89 날짜 간 최근접 감량 결과 대조','',
           '불일치율은 가장 가까운 앞 구간 사례와 확인 사례의 공동 개선 여부가 다른 비율이다. '
           'Brier는 서로 다른 날짜의 최근접 5건에서 얻은 성공 비율과 확인 정답 간 제곱오차다. 두 지표 모두 낮을수록 좋다. '
           '후보는 각 날짜 안에서 평균한 뒤 적격 날짜 및 seed에 같은 가중치를 준다.','',
           '| 질의 범위 | seed | 후보 | 날짜 | 불일치 관측 | 불일치 이력 추가 | 무작위 기대 | Brier 관측 | Brier 이력 추가 | 상수 Brier |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group,label in [('supported','학습 거리 95백분위 이내'),('close','학습 거리 중앙값 이내'),('all','전체')]:
        for r in records:
            a,b=[r['metrics'][group][m] for m in ('current','history')]
            lines.append(f"| {label} | {r['seed']} | {a['n']} | {a['days']} | {a['mismatch']:.1%} | "
                         f"{b['mismatch']:.1%} | {a['chance_mismatch']:.1%} | {a['brier']:.4f} | "
                         f"{b['brier']:.4f} | {a['chance_brier']:.4f} |")
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('/home/jio/fcsrl_src/results/e88/main'))
    args=p.parse_args()
    report(args.root,args.dest,args.source)
