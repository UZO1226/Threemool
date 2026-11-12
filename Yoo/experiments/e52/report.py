"""e52 집계. 기준(kappa 0)의 seed 0, 1 은 e45 결정적 평가(e44 최종 정책, 같은 환경과 평가 함수)를 쓴다.

    python report.py --root /home/jio/fcsrl_src/results/e52/main --out .
"""
import argparse
import json
from pathlib import Path

import numpy as np

E45 = Path('/home/jio/fcsrl_src/results/e45/main/jobs')
NAMES = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO-Lag'}
KEYS = ('air_m3', 'nh4_cost', 'do_cost', 'nh4_excess_mg_h_L', 'worst_day_nh4_cost')


def load(root, m, k, s):
    if k == 0 and s in (0, 1):
        get = lambda sp: json.loads((E45 / f'{m}_padded_s{s}_{sp}_det' / 'metrics.json').read_text())['summary']  # noqa: E731
        return {'val': get('val'), 'train': get('train'), 'source': 'e44 정책, e45 결정적 평가'}
    d = json.loads((root / 'runs' / m / f'k{k:g}' / f'seed_{s}' / 'metrics.json').read_text())
    return {'val': d['val_final'], 'train': d['train_final'], 'source': 'e52'}


def main(root, out):
    rows = []
    for m in NAMES:
        for k in (0, 10):
            for s in (0, 1, 2):
                r = load(root, m, k, s)
                rows.append({'method': m, 'kappa': k, 'seed': s, 'source': r['source'],
                             **{f'val_{x}': r['val'][x] for x in KEYS}, **{f'train_{x}': r['train'][x] for x in KEYS[:3]},
                             'val_both_limits': bool(r['val']['nh4_cost'] <= 6 and r['val']['do_cost'] <= 6),
                             'gap_nh4': r['val']['nh4_cost'] - r['train']['nh4_cost']})
    md = ['검증 40일 x 구성원 5개 = 200 에피소드와 학습 188일 x 5 = 940 에피소드, 결정적 행동, 마지막 정책, 평가 비용은 비관 없이 원래대로. '
          'NH4, DO 비용은 하루 위반 스텝 수 (95 스텝 중), 한도 각 6.', '',
          '| 방법 | kappa | seed | 검증 공기량 | 검증 NH4 | 검증 DO | 검증 NH4 초과 적분 | 최악 날짜 NH4 | 학습 NH4 | 격차 (검증 − 학습 NH4) | 출처 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        md.append(f"| {NAMES[r['method']]} | {r['kappa']} | {r['seed']} | {r['val_air_m3']:,.0f} | {r['val_nh4_cost']:.2f} | "
                  f"{r['val_do_cost']:.2f} | {r['val_nh4_excess_mg_h_L']:.2f} | {r['val_worst_day_nh4_cost']:.1f} | "
                  f"{r['train_nh4_cost']:.2f} | {r['gap_nh4']:.2f} | {r['source']} |")
    md += ['', '| 방법 | kappa | 검증 공기량 | 검증 NH4 | 검증 DO | 격차 | 검증 두 한도 충족 seed |', '| --- | --- | --- | --- | --- | --- | --- |']
    verdict = {}
    for m in NAMES:
        g = {k: [r for r in rows if r['method'] == m and r['kappa'] == k] for k in (0, 10)}
        for k, rs in g.items():
            f = lambda key: (np.mean([r[key] for r in rs]), np.std([r[key] for r in rs]))  # noqa: E731
            md.append(f"| {NAMES[m]} | {k} | {f('val_air_m3')[0]:,.0f} ± {f('val_air_m3')[1]:,.0f} | {f('val_nh4_cost')[0]:.2f} ± {f('val_nh4_cost')[1]:.2f} | "
                      f"{f('val_do_cost')[0]:.2f} ± {f('val_do_cost')[1]:.2f} | {f('gap_nh4')[0]:.2f} | {sum(r['val_both_limits'] for r in rs)}/3 |")
        d_nh4 = [b['val_nh4_cost'] - a['val_nh4_cost'] for a, b in zip(g[0], g[10])]
        d_gap = [b['gap_nh4'] - a['gap_nh4'] for a, b in zip(g[0], g[10])]
        d_air = [100 * (b['val_air_m3'] / a['val_air_m3'] - 1) for a, b in zip(g[0], g[10])]
        d_do = [b['val_do_cost'] - a['val_do_cost'] for a, b in zip(g[0], g[10])]
        verdict[m] = {'d_val_nh4_by_seed': d_nh4, 'd_gap_by_seed': d_gap, 'd_air_pct_by_seed': d_air, 'd_val_do_by_seed': d_do,
                      'H1': bool(sum(x < 0 for x in d_nh4) >= 2 and np.mean(d_nh4) < 0),
                      'H2': bool(np.mean(d_gap) < 0),
                      'air_up_over_5pct': bool(np.mean(d_air) > 5),
                      'all_three_lower_mean': bool(np.mean(d_nh4) < 0 and np.mean(d_air) < 0 and np.mean(d_do) < 0)}
    md += ['', '| 방법 | seed별 검증 NH4 차이 (kappa 10 − 0) | seed별 공기량 차이 % | seed별 DO 차이 | H1 | H2 격차 감소 | 공기량 +5% 초과 | 세 지표 평균 모두 감소 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for m, v in verdict.items():
        j = lambda xs, f='{:+.2f}': ', '.join(f.format(x) for x in xs)  # noqa: E731
        md.append(f"| {NAMES[m]} | {j(v['d_val_nh4_by_seed'])} | {j(v['d_air_pct_by_seed'])} | {j(v['d_val_do_by_seed'])} | "
                  f"{'지지' if v['H1'] else '지지 안 됨'} | {'예' if v['H2'] else '아니오'} | {'예' if v['air_up_over_5pct'] else '아니오'} | "
                  f"{'예' if v['all_three_lower_mean'] else '아니오'} |")
    (out / 'summary.json').write_text(json.dumps({'rows': rows, 'verdict': verdict}, indent=1, ensure_ascii=False))
    (out / 'results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.out)
