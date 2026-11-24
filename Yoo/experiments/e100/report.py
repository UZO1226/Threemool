"""Generate the e100 tables and audit each margin and mask from the saved arrays."""
import argparse
import json
from pathlib import Path

import numpy as np


def improves(delta):
    return (delta[..., 0] < -1e-8) & (delta[..., 1:] <= 1e-8).all(axis=-1)


def export(root, dest):
    rows = json.loads((root/'summary.json').read_text())
    assert [s['seed'] for s in rows] == list(range(5))
    lines = ['# e100 공기 오차 보정 후 첫 감량 결과', '',
             '기본 FCSRL 대비 6시간 변화다. 공기는 m³, 위반은 15분 스텝 수, NH4 초과 적분은 mg h/L다. '
             '선택 수와 공동 개선 창 수는 합계이고 제어 지표는 날짜와 seed에 같은 가중치를 준 평균이다.', '']
    for name, title in [('supported','거리 조건 안'), ('all','전체 비중복 창')]:
        gs = [s['groups'][name] for s in rows]
        lines += [f'## {title}', '', f"전체 {sum(g['n'] for g in gs):,}개 창 중 후보 {sum(g['candidates'] for g in gs)}개.", '',
                  '| 감량 선택 | 횟수 | 공동 개선 창 | 공기 | NH4 위반 | DO 위반 | NH4 초과 적분 | 공동 개선 seed |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for key, label in [('base','기본 FCSRL'),('original','e96'),('calibrated','보정'),
                           ('random','후보 내 동일 횟수 무작위 기대'),('retained','e96 선택 내 동일 횟수 무작위 기대')]:
            s = [g['outcomes'][key] for g in gs]
            delta = np.array([v['delta'] for v in s])
            vals = ' | '.join(f'{v:+.5f}' for v in delta.mean(0))
            lines.append(f"| {label} | {sum(v['n'] for v in s)} | {sum(v['hits'] for v in s):.2f} | {vals} | {improves(delta).sum()}/5 |")
        lines += ['', '| seed | 보정 창 수 (두 블록) | 해당 날짜 수 | 여유 (m³) | 후보 잔차가 여유 이하 | 원래 선택 | 원래 선택 중 최대 예상 절감 (m³) | 보정 선택 | 공기 변화 | NH4 위반 | DO 위반 | NH4 적분 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r, g in zip(rows, gs):
            s = g['outcomes']['calibrated']
            with np.load(root/f"cut{r['seed']}.npz") as z:
                d, e = z['cal_day'], z['cal_residual']
                # Each distinct date contributes one empirical distribution.
                q = r['margin']
                cdf = lambda less: np.mean([less[d == day].mean() for day in np.unique(d)])
                assert q >= 0 and cdf(e <= q) >= .8-1e-12 and (q == 0 or cdf(e < q) <= .8+1e-12)
                mask = z['supported'] if name == 'supported' else np.ones(len(z['first']), bool)
                old = z['first'] & improves(z['pred'])
                take = old & (z['pred'][:,0]+q < -1e-8)
                np.testing.assert_array_equal(take, z['calibrated'])
                assert (take & mask).sum() == s['n']
                savings = -z['pred'][old & mask, 0]
                largest = f'{savings.max():,.2f}' if len(savings) else '해당 없음'
                n = [b['cal'] for b in r['blocks']]
                counts = [b['dates'] for b in r['blocks']]
            vals = ' | '.join(f'{v:+.5f}' for v in s['delta'])
            lines.append(f"| {r['seed']} | {n[0]} + {n[1]} | {counts[0]} + {counts[1]} | {q:,.2f} | {g['coverage']:.1%} | {g['outcomes']['original']['n']} | {largest} | {s['n']} | {vals} |")
        lines.append('')
    dest.write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--dest', required=True, type=Path)
    args = p.parse_args()
    export(args.root, args.dest)
