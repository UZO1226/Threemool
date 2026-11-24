"""Audit saved paired branches and present the predeclared seed/block comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

LABELS = dict(base='기본', half='고정 250', full='고정 500', adaptive='현재 NH4로 크기 결정',
              random='같은 크기 횟수 무작위 기대', oracle='모의 결과를 아는 사후 선택')


def joint(v):
    return (v[...,0] < -1e-8) & (v[...,1:] <= 1e-8).all(axis=-1)


def report(root, dest):
    records = json.loads((root/'summary.json').read_text())
    lines = ['# e102 첫 감량 크기별 6시간 결과', '',
             '각 값은 기본 FCSRL 대비 변화다. 음수면 공기를 아끼거나 수질 지표가 개선됐다. 주 집합 창을 날짜별로 평균한 뒤 seed에 같은 가중치를 줬다.', '']
    extra = {1: [], 2: []}
    for r in records:
        with np.load(root/f"size{r['seed']}.npz") as z:
            start = 70 if r['block']==1 else 105
            days = z['keys'][:,0]
            mask = (days>=start)&(days<start+35)
            t, n, days = z['totals'][mask], z['nh4'][mask], days[mask]
            a = np.select([n>=4, n>2],[0,1],default=2)
            np.testing.assert_array_equal(np.bincount(a, minlength=3),r['levels'])
            deltas = t-t[:,:1]
            # Post hoc: quantify how much the middle action adds to the per-window oracle.
            permitted = np.where((deltas[:,:,1:] <= 1e-8).all(2), deltas[:,:,0], np.inf)
            gain = permitted[:,[0,2]].min(1) - permitted.min(1)
            extra[r['block']].append(np.mean([gain[days==d].mean() for d in np.unique(days)]))
            for mode in LABELS:
                daily = []
                for d in np.unique(days):
                    ix = np.flatnonzero(days==d)
                    if mode == 'random':
                        v = np.mean([deltas[ix, j].mean(0) for j in a[ix]],0)
                    else:
                        if mode == 'oracle':
                            take = [min((j for j in range(3) if np.all(deltas[i,j,1:]<=1e-8)),
                                        key=lambda j: deltas[i,j,0]) for i in ix]
                        elif mode == 'adaptive':
                            take = a[ix]
                        else:
                            take = {'base':0,'half':1,'full':2}[mode]
                        v = deltas[ix,take].mean(0)
                    daily.append(v)
                np.testing.assert_allclose(np.mean(daily,0), r['outcomes'][mode], atol=1e-8,rtol=0)
    wins = {k: np.ones(5,bool) for k in [*LABELS,'adaptive_minus_random']}
    for b in (1,2):
        rs = sorted((r for r in records if r['block']==b), key=lambda r:r['seed'])
        assert [r['seed'] for r in rs] == list(range(5))
        lines += [f'## 블록 {b}', '', '| 방식 | 공기 (m³) | NH4 위반 (스텝) | DO 위반 (스텝) | NH4 초과 적분 (mg h/L) | 공동 개선 seed |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        vs = {k: np.array([r['outcomes'][k] for r in rs]) for k in LABELS}
        vs['adaptive_minus_random'] = vs['adaptive']-vs['random']
        for mode, v in vs.items():
            good = joint(v)
            wins[mode] &= good
            fmt = ' | '.join(f'{a:+.5f}' for a in v.mean(0))
            lines.append(f"| {LABELS.get(mode,'적응형 빼기 무작위')} | {fmt} | {good.sum()}/5 |")
        lines += ['', '| seed | 창 | 적격일 | 0 / 250 / 500 선택 횟수 | 적응형 공기 | NH4 위반 | DO 위반 | NH4 초과 적분 |',
                  '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
        for r in rs:
            counts = ' / '.join(map(str,r['levels']))
            fmt = ' | '.join(f'{x:+.5f}' for x in r['outcomes']['adaptive'])
            lines.append(f"| {r['seed']} | {r['n']} | {r['dates']} | {counts} | {fmt} |")
        lines.append('')
    lines += ['## 같은 seed가 두 블록 모두에서 공동 개선한 횟수', '']
    for mode, val in wins.items():
        lines.append(f"- {LABELS.get(mode,'적응형 빼기 무작위')} {val.sum()}/5")
    lines += ['', '## 사후 보조 분해', '',
              '결과를 본 뒤 같은 창에서 0 또는 500만 허용한 사후 선택과 비교했다. '
              '250을 추가했을 때 창별 수질 비악화 제약 아래 가능한 추가 공기 절감(두 블록 순서, m³)은 '
              + ', '.join(f'{np.mean(extra[b]):.2f}' for b in (1,2))+'였다.']
    dest.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('dest', type=Path)
    args = p.parse_args()
    report(args.root,args.dest)
