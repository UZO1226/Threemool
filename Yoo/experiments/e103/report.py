"""Check pairwise transfers directly and tabulate block and seed results."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(root, r):
    with np.load(root/f"transfer{r['seed']}.npz") as z:
        start = 70 if r['block']==1 else 105
        keys = z['keys']
        use = (keys[:,0]>=start)&(keys[:,0]<start+35)
        ds = keys[use,0]
        totals = z['totals'][use]
        count = np.zeros(3,int)
        rows = []
        for m in totals:
            dif = m-m[:,:1]
            allowed = (dif[:,:,0] < -1e-8)&(dif[:,:,1:]<=1e-8).all(2)
            picks = [min([0]+[i for i in (1,2) if allowed[j,i]],key=lambda i: m[j,i,0]) for j in range(5)]
            count += [max(picks)==0,min(picks)>0 and len(set(picks))==1, allowed[:,1:].all(0).any()]
            values = []
            for a,pick in enumerate(picks):
                for b in range(5):
                    if a != b:
                        values.append([pick>0,allowed[b,pick],*dif[b,pick]])
            rows.append(np.mean(values,0))
        np.testing.assert_array_equal(count,[r[k] for k in ('all_zero','unanimous_cut','common')])
        per_day = np.mean([np.array(rows)[ds==d].mean(0) for d in np.unique(ds)],0)
        g = r['cross']
        np.testing.assert_allclose(per_day[2:],g['delta'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(per_day[:2], [g['acted'], g['acted']*(g['joint'] or 0)], atol=1e-8,rtol=0)


def pct(x):
    return '계산 불가' if x is None else f'{x:.1%}'


def render(root,dest):
    records = json.loads((root/'summary.json').read_text())
    for r in records:
        audit(root,r)
    lines = ['# e103 모델 사이의 첫 감량 전이 결과', '',
             '크기 일치와 전이율은 날짜에 같은 비중을 준 seed별 비율을 평균했다. '
             '조건부 비율은 결정 모델이 250 또는 500을 골랐을 때만 계산하며 계산 가능 seed 수를 표시한다. '
             '공기와 수질 차이는 0 선택도 포함한 6시간 창 평균이다.', '']
    for b in (1,2):
        rs = sorted((r for r in records if r['block']==b),key=lambda r:r['seed'])
        assert [r['seed'] for r in rs]==list(range(5))
        n,z,c,anycut = [sum(r[k] for r in rs) for k in ('n','all_zero','unanimous_cut','common')]
        lines += [f'## 블록 {b}', '', f'창 {n}개 중 다섯 모델 모두 감량 없음 {z}개, '
                  f'같은 양의 감량이 최선 {c}개, 그 밖에 최선 선택 불일치 {n-z-c}개다. '
                  f'최선 순위와 무관하게 다섯 모델에 모두 유리한 양의 감량이 있는 창은 {anycut}개다.', '',
                  '| 결정 모델 및 채점 모델 | 감량 선택 비율 | 전체 최선 일치 | 감량 시 최선 일치 | 감량 시 공기 감소 | 감량 시 세 수질 비악화 | 감량 시 공동 개선 | 계산 seed |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        modes = [('cross','모든 다른 모델쌍'),('origin','창 출처 모델에서 다른 모델로')]
        for key,label in modes:
            data = [r[key] for r in rs]
            avgs = []
            for k in ('acted','same','best','air','quality','joint'):
                vals = [g[k] for g in data if g[k] is not None]
                avgs.append(pct(np.mean(vals) if vals else None))
            lines.append(f"| {label} | "+' | '.join(avgs)+f' | {len(vals)}/5 |')
        lines += ['', '| 모델쌍 | 공기 (m³) | NH4 위반 (스텝) | DO 위반 (스텝) | NH4 초과 적분 (mg h/L) | 공동 개선 seed |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for key,label in modes:
            dif = np.array([r[key]['delta'] for r in rs])
            ok = (dif[:,0]<-1e-8)&(dif[:,1:]<=1e-8).all(1)
            lines.append(f'| {label} | '+' | '.join(f'{v:+.5f}' for v in dif.mean(0))+f' | {ok.sum()}/5 |')
        lines += ['', '| seed | 창 | 감량 전이 비교 수 | 감량 공동 개선 전이율 | 전이 공기 변화 | NH4 위반 | DO 위반 | NH4 초과 적분 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rs:
            g = r['cross']
            lines.append(f"| {r['seed']} | {r['n']} | {g['cases']} | {pct(g['joint'])} | "+
                         ' | '.join(f'{v:+.5f}' for v in g['delta'])+' |')
        signs = np.mean([r['sign_match'] for r in rs],0)
        lines += ['', '공기 부피 순서의 두 모델 간 일치율은 0 대 250, 0 대 500, 250 대 500 순으로 '
                  + ', '.join(map(pct,signs))+'다.', '']
    lines += ['## 사전 반복 기준', '']
    counts = []
    for seed in range(5):
        gs = [r['cross'] for r in records if r['seed']==seed]
        majority = all(g['joint'] is not None and g['joint']>.5 for g in gs)
        effects = all(g['delta'][0] < -1e-8 and max(g['delta'][1:]) <= 1e-8 for g in gs)
        counts.append((majority,effects))
    majority, effects = np.sum(counts,axis=0)
    lines += [f'두 블록 모두 감량 전이 성공률이 절반을 넘은 seed는 {majority}/5, '
              f'두 블록 모두 다른 모델에서 기본 대비 네 지표를 함께 개선한 seed는 {effects}/5였다.', '']
    dest.write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('dest',type=Path)
    args=p.parse_args()
    render(args.root,args.dest)
