"""e31 2단계 결과: 같은 배분 방식의 기록 운전 재생 대비 차이

    python report.py --prepared /home/jio/fcsrl_src/results/e31/prepared --runs /home/jio/fcsrl_src/results/e31/stage2 --out .
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import make_env, rollout  # noqa: E402

NAMES = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO-Lag'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base = {}
    for frac in ('recorded', 'fixed'):
        splits, env = make_env(a.prepared, frac)
        rows = rollout(env, splits['val'], lambda x, e: e.historical_action())
        base[frac] = pd.DataFrame(rows).set_index(['day', 'member'])
    recs = []
    for f in sorted(a.runs.glob('*/seed_*/metrics.json')):
        m = json.loads(f.read_text())
        ep = pd.DataFrame(json.loads((f.parent / 'val_episodes.json').read_text())).set_index(['day', 'member'])
        b = base[m['frac']].loc[ep.index]
        recs.append({'method': m['method'], 'frac': m['frac'], 'seed': m['seed'], 'air_m3': ep.air_m3.mean(),
                     'cost': ep.cost.mean(), 'nh4_pct': 100 * ep.nh4_steps.mean() / 95, 'do_pct': 100 * ep.do_steps.mean() / 95,
                     'd_air_pct': 100 * (ep.air_m3.mean() / b.air_m3.mean() - 1), 'd_cost': ep.cost.mean() - b.cost.mean(),
                     'within_limit_pct': 100 * (ep.cost <= 12).mean(), 'final_lambda': m.get('final_lambda')})
    r = pd.DataFrame(recs).sort_values(['frac', 'method', 'seed'])
    md = ['검증 46일 x GBM 구성원 5개 = 230 에피소드. 기준은 같은 배분 방식으로 기록 총 공기량을 재생한 결과다.', '',
          '| 배분 | 기록 운전 재생 공기량 (Nm3/일) | 재생 하루 비용 | 재생 NH4 위반 | 재생 DO 위반 |', '| --- | --- | --- | --- | --- |']
    for frac, b in base.items():
        md.append(f"| {frac} | {b.air_m3.mean():,.0f} | {b.cost.mean():.2f} | {100 * b.nh4_steps.mean() / 95:.2f}% | "
                  f"{100 * b.do_steps.mean() / 95:.2f}% |")
    md += ['', '| 배분 | 방법 | seed | 공기량 (Nm3/일) | 재생 대비 공기량 | 하루 비용 | 재생 대비 비용 | NH4 위반 | DO 위반 | 한도 12 이하 에피소드 | 마지막 lambda |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for _, x in r.iterrows():
        md.append(f"| {x.frac} | {NAMES[x.method]} | {x.seed} | {x.air_m3:,.0f} | {x.d_air_pct:+.2f}% | {x.cost:.2f} | "
                  f"{x.d_cost:+.2f} | {x.nh4_pct:.2f}% | {x.do_pct:.2f}% | {x.within_limit_pct:.1f}% | {x.final_lambda:.2f} |")
    g = r.groupby(['frac', 'method'])
    md += ['', '| 배분 | 방법 | seed 수 | 재생 대비 공기량 평균 ± 표준편차 | 재생 대비 비용 평균 ± 표준편차 | 둘 다 개선한 seed |',
           '| --- | --- | --- | --- | --- | --- |']
    for (frac, meth), x in g:
        both = int(((x.d_air_pct < 0) & (x.d_cost <= 0)).sum())
        md.append(f"| {frac} | {NAMES[meth]} | {len(x)} | {x.d_air_pct.mean():+.2f}% ± {x.d_air_pct.std():.2f} | "
                  f"{x.d_cost.mean():+.2f} ± {x.d_cost.std():.2f} | {both}/{len(x)} |")
    (a.out / 'results.md').write_text('\n'.join(md) + '\n')
    r.to_csv(a.out / 'results_per_seed.csv', index=False)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
