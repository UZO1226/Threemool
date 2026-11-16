"""e50 결과 집계: 방법별 seed 평균 ± 표준편차(ddof=0, Park/RESULT.md 와 같은 방식)와 사전 등록 판정

    python report.py --root /home/jio/fcsrl_src/results/e50/main --out .
"""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO-Lag', 'fcsrl_actor': 'FCSRL + Actor 보정 (Yoo 확장)'}
E30 = {'ppo': 11.45, 'fcsrl': 16.58, 'fcsrl_actor': 13.68}  # e30 (scikit-learn 1.7.2 모델) 평균 비용
KEYS = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')


def main(root, out):
    manifest = json.loads((root / 'manifest.json').read_text())
    runs = {m: {s: json.loads((root / 'runs' / m / f'seed_{s}' / 'metrics.json').read_text())
                for s in manifest['seeds']} for m in METHODS}
    for m, by_seed in runs.items():
        assert all(r['steps'] == manifest['steps_per_run'] for r in by_seed.values()), m
    val = {m: {s: r['val_final'] for s, r in by_seed.items()} for m, by_seed in runs.items()}
    summary = {}
    for m, v in val.items():
        summary[m] = {k: {'mean': float(np.mean([x[k] for x in v.values()])), 'sd': float(np.std([x[k] for x in v.values()]))}
                      for k in KEYS}
        summary[m]['seeds_mean_cost_within_12'] = int(sum(x['cost'] <= 12 for x in v.values()))
        summary[m]['reward_per_day'] = {'mean': -summary[m]['air_m3']['mean'] / 0.25 / 9000}
    seeds = manifest['seeds']
    d_cost = [val['fcsrl_actor'][s]['cost'] - val['fcsrl'][s]['cost'] for s in seeds]
    d_air = [100 * (val['fcsrl_actor'][s]['air_m3'] / val['fcsrl'][s]['air_m3'] - 1) for s in seeds]
    h1 = {'cost_diff_by_seed': d_cost, 'air_diff_pct_by_seed': d_air,
          'seeds_cost_lower': int(sum(d < 0 for d in d_cost)), 'mean_cost_diff': float(np.mean(d_cost)),
          'mean_paired_air_diff_pct': float(np.mean(d_air)),
          'mean_air_diff_pct': float(100 * (summary['fcsrl_actor']['air_m3']['mean'] / summary['fcsrl']['air_m3']['mean'] - 1))}
    h1['supported'] = bool(h1['seeds_cost_lower'] >= 4 and h1['mean_cost_diff'] < 0 and h1['mean_air_diff_pct'] <= 2)
    order_now = sorted(METHODS, key=lambda m: summary[m]['cost']['mean'])
    order_e30 = sorted(METHODS, key=lambda m: E30[m])
    result = {'summary': summary, 'per_seed': val, 'H1_actor_vs_fcsrl': h1,
              'Q2_cost_order': {'e50': order_now, 'e30': order_e30, 'same': order_now == order_e30}}
    (out / 'summary.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    md = ['검증 40일 x 구성원 5개 = 200 에피소드, 결정적 행동, 마지막 정책, seed 0 ~ 4 평균 ± 표준편차(ddof=0). '
          '학습 예산 401,280 전이. 기준 대리 모델(scikit-learn 1.9.1) 환경.', '',
          '| 방법 | 공기량 (m3/23.75h) | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 이하 에피소드 % | 평균 비용 <= 12 seed | e30 평균 비용 (1.7.2 모델) |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for m, name in METHODS.items():
        s = summary[m]
        md.append(f"| {name} | {s['air_m3']['mean']:,.0f} ± {s['air_m3']['sd']:,.0f} | {s['cost']['mean']:.2f} ± {s['cost']['sd']:.2f} | "
                  f"{s['nh4_viol_pct']['mean']:.2f} ± {s['nh4_viol_pct']['sd']:.2f} | {s['do_viol_pct']['mean']:.2f} ± {s['do_viol_pct']['sd']:.2f} | "
                  f"{s['days_within_limit_pct']['mean']:.1f} | {s['seeds_mean_cost_within_12']}/5 | {E30[m]:.2f} |")
    md += ['', '| seed | ' + ' | '.join(f'{METHODS[m]} 비용 (공기량)' for m in METHODS) + ' | Actor 보정 − FCSRL 비용 | 공기량 차이 % |',
           '| --- | ' + ' | '.join('---' for _ in METHODS) + ' | --- | --- |']
    for i, s in enumerate(seeds):
        md.append(f'| {s} | ' + ' | '.join(f"{val[m][s]['cost']:.2f} ({val[m][s]['air_m3']:,.0f})" for m in METHODS)
                  + f' | {d_cost[i]:+.2f} | {d_air[i]:+.2f} |')
    md += ['', f"H1 (Actor 보정이 FCSRL 보다 비용이 낮은 seed {h1['seeds_cost_lower']}/5, 평균 비용 차이 {h1['mean_cost_diff']:+.2f}, "
           f"평균 공기량 차이 {h1['mean_air_diff_pct']:+.2f}%): {'지지' if h1['supported'] else '지지 안 됨'}",
           f"Q2 비용 순위: e50 {' < '.join(METHODS[m] for m in order_now)}, e30 {' < '.join(METHODS[m] for m in order_e30)} "
           f"({'같음' if result['Q2_cost_order']['same'] else '다름'})"]
    (out / 'results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.out)
