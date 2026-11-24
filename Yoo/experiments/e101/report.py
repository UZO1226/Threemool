"""Tabulate chronological ranking results and their denominators."""
import argparse
import json
from pathlib import Path
from statistics import mean


def number(value):
    return '계산 불가' if value is None else f'{value:+.3f}'


def render(records):
    lines = ['# e101 절감 순위와 같은 날짜 상위 절반의 성과', '',
             '예측과 모의 절감의 Spearman 상관은 seed 및 블록 안의 창을 사용한다. '
             '날짜별 상위 절반 및 무작위의 성과는 두 후보 이상 있는 날짜만 같은 비중으로 평균했다. '
             '공기 절감(m³)은 기본보다 덜 쓰면 양수다. 세 수질 변화는 기본보다 악화하면 양수다.', '']
    for name, title in [('quality','세 예측 수질 조건 통과'), ('all','수질 선별 전')]:
        lines += [f'## {title}', '',
                  '| 블록 | seed | 후보 창 | 후보 없는 날 | 1개인 날 | 비교 가능일 | 전부 동점인 비교일 | 선택 수 | 순위 상관 | 무작위 대비 추가 절감 (m³) |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        repeated = set(range(5))
        for start in (70,105):
            rs = [r for r in records if r['start'] == start]
            ok = set()
            for r in rs:
                g = r['groups'][name]
                assert g['empty']+g['singleton']+g['dates'] == 35
                if all(g[k] is not None and g[k] > 0 for k in ('rho','lift')):
                    ok.add(r['seed'])
                fields = [1 if start == 70 else 2, r['seed'], g['n'], g['empty'], g['singleton'],
                          g['dates'], g['ties'], g['slots'], number(g['rho']), number(g['lift'])]
                lines.append('| '+' | '.join(map(str,fields))+' |')
            repeated &= ok
        lines += ['', f'두 블록에서 상관과 추가 절감이 모두 양수인 seed는 {len(repeated)}/5였다.', '',
                  '| 블록 | 방식 | 계산 seed | 절감 (m³) | NH4 위반 (스텝) | DO 위반 (스텝) | NH4 초과 (mg h/L) |',
                  '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
        for i,start in enumerate((70,105),1):
            groups = [r['groups'][name] for r in records if r['start'] == start]
            for key, label in [('top','상위 절반'),('random','동일 횟수 무작위 기대')]:
                rows = [g[key] for g in groups if g[key] is not None]
                values = [number(mean(col)) for col in zip(*rows)] if rows else ['계산 불가']*4
                lines.append(f'| {i} | {label} | {len(rows)}/5 | '+' | '.join(values)+' |')
        for i,start in enumerate((70,105),1):
            gs = [r['groups'][name] for r in records if r['start'] == start]
            stats = []
            for field in ('rho','lift'):
                vals = [g[field] for g in gs if g[field] is not None]
                stats.append(f"{number(mean(vals) if vals else None)} (양수 {sum(v>0 for v in vals)}/{len(vals)})")
            lines += ['', f'블록 {i}의 seed 평균 상관은 {stats[0]}, 무작위 대비 추가 절감은 {stats[1]} m³다.']
        lines.append('')
    return '\n'.join(lines).rstrip()+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('summary', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.output.write_text(render(json.loads(args.summary.read_text())))
