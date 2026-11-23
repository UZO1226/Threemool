"""Recalculate episode metrics from the saved trajectories and compare paired seeds."""
import argparse
import json
from pathlib import Path

import numpy as np


def report(root, dest):
    modes = ['base','pulse','recover']
    metrics = ['air','nh','do','excess','limit_pct','raw_air','added','removed']
    records = json.loads((root/'records.json').read_text())
    rows = {(r['mode'], r['seed']):r for r in records}
    assert len(records) == len(rows) == 15
    values = {}
    for m in modes:
        samples = []
        for seed in range(5):
            r = rows[m,seed]
            with np.load(root/f'{seed}_{m}.npz') as z:
                assert z['states'].shape == (200,95,5) and len(set(z['dates'])) == 40
                states = z['states']
                nh = np.count_nonzero(states[:,:,0] > 4, axis=1)
                do = (states[:,:,2:] > 2).any(2).sum(1)
                delta = z['delta']
                block = [z['air'].sum()/800, nh.mean(), do.mean(),
                         np.clip(states[:,:,0]-4,0,None).sum()/800,
                         100*np.mean(nh+do<=12), z['raw'].sum()/800,
                         np.maximum(delta,0).sum()/800,-np.minimum(delta,0).sum()/800]
                np.testing.assert_allclose(block, [r[k] for k in metrics],rtol=0,atol=1e-8)
                samples.append(block)
        values[m] = np.array(samples)
    lines = ['# e81 겨울 하루 운전 결과','',
             '각 수치는 40일과 5개 GBM 구성원의 95스텝 평균을 구한 뒤 정책 seed 5개를 같은 비중으로 집계했다. 기본은 base, 증량 단독은 pulse, 증량 후 회수는 recover다.','',
             '| 방식 | 공기 m³/23.75h | NH4 위반 | DO 위반 | NH4 초과 mg h/L | 합산 한도 12 충족률 % | 직접 추가 m³ | 직접 회수 m³ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in modes:
        lines.append('| '+m+' | '+' | '.join(f'{x:.4f}' for x in values[m].mean(0)[[0,1,2,3,4,6,7]])+' |')
    for left,right in [('base','pulse'),('pulse','recover'),('base','recover')]:
        delta = values[right]-values[left]
        lines += ['',f'## {right} 빼기 {left}','',
                  '| seed | 공기 차 | NH4 위반 차 | DO 위반 차 | NH4 초과 차 | 기본 명령 부피 차 | 직접 보정 부피 차 | 공동 개선 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        rows_delta = np.vstack((delta,delta.mean(0)))
        wins = (rows_delta[:,0]<-1e-8) & (rows_delta[:,1:4]<=1e-8).all(1)
        for seed,d in enumerate(rows_delta):
            direct = d[6]-d[7]
            assert abs(d[0]-d[5]-direct)<1e-8
            mark = ('충족' if wins[seed] else '미달')
            if seed == 5:
                mark += f', {sum(wins[:5])}/5 seed'
            lines.append('| '+(str(seed) if seed<5 else '평균')+' | '+
                         ' | '.join(f'{x:+.4f}' for x in [*d[:4],d[5],direct])+' | '+mark+' |')
    lines += ['','## 회수 발생 범위','',
              '분모는 seed별 날짜와 구성원 200조합이다.','',
              '| seed | 증량 에피소드 | 1회 이상 회수 | 2회 모두 회수 |','| --- | ---: | ---: | ---: |']
    for seed in range(5):
        r = rows['recover',seed]
        lines.append(f"| {seed} | {r['pulsed']} | {r['cut_once_or_more']} | {r['cut_twice']} |")
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    report(args.root,args.dest)
