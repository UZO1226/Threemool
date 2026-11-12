"""e53 집계. 주 비교는 e52 kappa 10 (구성원 불일치 여유) 대 e53 delta 0.45 (고정 여유), seed 짝.
kappa 0 기준은 e52 와 같은 출처(seed 0, 1 은 e45 결정적 평가, seed 2 는 e52)를 참고로 함께 적는다.

    python report.py --root /home/jio/fcsrl_src/results/e53/main --e52 /home/jio/fcsrl_src/results/e52/main --out .
"""
import argparse
import json
from pathlib import Path

import numpy as np

E45 = Path('/home/jio/fcsrl_src/results/e45/main/jobs')
NAMES = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO-Lag'}
KEYS = ('air_m3', 'nh4_cost', 'do_cost', 'nh4_excess_mg_h_L', 'worst_day_nh4_cost')
DELTA = 0.45
COND = ('k0', 'k10', 'd0.45')
LABEL = {'k0': '기준 (여유 없음)', 'k10': '불일치 여유 kappa 10 (e52)', 'd0.45': '고정 여유 0.45 (e53)'}


def load(root, e52, m, c, s):
    if c == 'k0' and s in (0, 1):
        get = lambda sp: json.loads((E45 / f'{m}_padded_s{s}_{sp}_det' / 'metrics.json').read_text())['summary']  # noqa: E731
        return get('val'), get('train')
    base = e52 if c.startswith('k') else root
    d = json.loads((base / 'runs' / m / c / f'seed_{s}' / 'metrics.json').read_text())
    return d['val_final'], d['train_final']


def main(root, e52, out):
    rows = []
    for m in NAMES:
        for c in COND:
            for s in (0, 1, 2):
                v, t = load(root, e52, m, c, s)
                rows.append({'method': m, 'cond': c, 'seed': s, **{f'val_{x}': v[x] for x in KEYS},
                             **{f'train_{x}': t[x] for x in KEYS[:3]},
                             'val_both_limits': bool(v['nh4_cost'] <= 6 and v['do_cost'] <= 6),
                             'gap_nh4': v['nh4_cost'] - t['nh4_cost']})
    md = ['검증 40일 x 구성원 5개 = 200 에피소드와 학습 188일 x 5 = 940 에피소드, 결정적 행동, 마지막 정책, 평가 비용은 여유 없이 원래대로. '
          'NH4, DO 비용은 하루 위반 스텝 수 (95 스텝 중), 한도 각 6.', '',
          '| 방법 | 조건 | seed | 검증 공기량 | 검증 NH4 | 검증 DO | 검증 NH4 초과 적분 | 최악 날짜 NH4 | 학습 공기량 | 학습 NH4 | 격차 (검증 − 학습 NH4) |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        md.append(f"| {NAMES[r['method']]} | {LABEL[r['cond']]} | {r['seed']} | {r['val_air_m3']:,.0f} | {r['val_nh4_cost']:.2f} | "
                  f"{r['val_do_cost']:.2f} | {r['val_nh4_excess_mg_h_L']:.2f} | {r['val_worst_day_nh4_cost']:.1f} | "
                  f"{r['train_air_m3']:,.0f} | {r['train_nh4_cost']:.2f} | {r['gap_nh4']:.2f} |")
    md += ['', '| 방법 | 조건 | 검증 공기량 | 검증 NH4 | 검증 DO | 학습 공기량 | 학습 NH4 | 격차 | 검증 두 한도 충족 seed |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    verdict = {}
    for m in NAMES:
        g = {c: [r for r in rows if r['method'] == m and r['cond'] == c] for c in COND}
        for c, rs in g.items():
            f = lambda key: (np.mean([r[key] for r in rs]), np.std([r[key] for r in rs]))  # noqa: E731
            md.append(f"| {NAMES[m]} | {LABEL[c]} | {f('val_air_m3')[0]:,.0f} ± {f('val_air_m3')[1]:,.0f} | "
                      f"{f('val_nh4_cost')[0]:.2f} ± {f('val_nh4_cost')[1]:.2f} | {f('val_do_cost')[0]:.2f} ± {f('val_do_cost')[1]:.2f} | "
                      f"{f('train_air_m3')[0]:,.0f} | {f('train_nh4_cost')[0]:.2f} | {f('gap_nh4')[0]:.2f} | {sum(r['val_both_limits'] for r in rs)}/3 |")
        a, b = g['d0.45'], g['k10']  # 차이 = 불일치 여유 − 고정 여유
        d_nh4 = [y['val_nh4_cost'] - x['val_nh4_cost'] for x, y in zip(a, b)]
        d_air = [100 * (y['val_air_m3'] / x['val_air_m3'] - 1) for x, y in zip(a, b)]
        d_do = [y['val_do_cost'] - x['val_do_cost'] for x, y in zip(a, b)]
        d_gap = [y['gap_nh4'] - x['gap_nh4'] for x, y in zip(a, b)]
        nh4_route = sum(x < 0 for x in d_nh4) >= 2 and np.mean(d_nh4) < 0 and np.mean(d_air) <= 2
        air_route = sum(x < 0 for x in d_air) >= 2 and np.mean(d_air) <= -2 and np.mean(d_nh4) <= 0
        verdict[m] = {'d_val_nh4_by_seed': d_nh4, 'd_air_pct_by_seed': d_air, 'd_val_do_by_seed': d_do, 'd_gap_by_seed': d_gap,
                      'nh4_route': bool(nh4_route), 'air_route': bool(air_route), 'H1': bool(nh4_route or air_route),
                      'H2_gap_smaller': bool(np.mean(d_gap) < 0)}
    md += ['', '주 비교: 불일치 여유 (e52 kappa 10) − 고정 여유 (e53 delta 0.45), seed 짝.', '',
           '| 방법 | seed별 검증 NH4 차이 | seed별 공기량 차이 % | seed별 DO 차이 | 평균 NH4 차이 | 평균 공기량 차이 % | H1 불확실성 정보 기여 | H2 격차 더 작음 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for m, v in verdict.items():
        j = lambda xs: ', '.join(f'{x:+.2f}' for x in xs)  # noqa: E731
        route = '지지 (NH4 경로)' if v['nh4_route'] else '지지 (공기량 경로)' if v['air_route'] else '지지 안 됨'
        md.append(f"| {NAMES[m]} | {j(v['d_val_nh4_by_seed'])} | {j(v['d_air_pct_by_seed'])} | {j(v['d_val_do_by_seed'])} | "
                  f"{np.mean(v['d_val_nh4_by_seed']):+.2f} | {np.mean(v['d_air_pct_by_seed']):+.2f} | {route} | {'예' if v['H2_gap_smaller'] else '아니오'} |")
    (out / 'summary.json').write_text(json.dumps({'delta': DELTA, 'rows': rows, 'verdict': verdict}, indent=1, ensure_ascii=False))
    (out / 'results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--e52', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.e52, a.out)
