"""e18 결과 표와 그림

    python report.py --prepared <준비 폴더> --pilot <결과 폴더> --out .
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO-Lag'}
COLORS = {'ppo': '#ff7f0e', 'fcsrl': '#2ca02c'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.prepared / 'metadata.json').read_text())
    refs = json.loads((a.prepared / 'references.json').read_text())
    front = {float(k): v for k, v in json.loads((a.prepared / 'constant_setpoint_frontier.json').read_text()).items()}
    sps = sorted(front)
    fe = np.array([front[s]['aeration_kwh_per_day'] for s in sps])
    fn = np.array([front[s]['flow_weighted_nh4'] for s in sps])
    runs = {}
    for f in sorted(a.pilot.glob('*/seed_*/metrics.json')):
        m = json.loads(f.read_text())
        runs.setdefault(m['method'], []).append(m)
    dopi = meta['do_pi_training_days']['mean']
    lim_nh4 = dopi['flow_weighted_nh4']

    def frontier_nh4(e):
        return float(np.interp(e, fe, fn))  # fe 는 설정값과 함께 증가

    def frontier_energy(n):
        return float(np.interp(n, fn[::-1], fe[::-1]))

    md = ['## 학습 날짜 (건기 0 ~ 6일, 하루 reset) 의 마지막 정책', '',
          f'DO PI: 폭기 {dopi["aeration_kwh_per_day"]:.1f} kWh/일, NH4 {lim_nh4:.3f}, 초과 시간 {dopi["nh4_violation_percent"]:.2f}%', '',
          '| 방법 | seed | 폭기 kWh/일 | DO PI 대비 | 유량 가중 NH4 | 같은 NH4 의 고정 설정값 경계 에너지 | 경계 대비 | NH4 > 4 시간 | 마지막 lambda | 마지막 10회 정책 clip % | 가치 clip % |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for meth in ('ppo', 'fcsrl'):
        for m in sorted(runs.get(meth, []), key=lambda m: m['seed']):
            f = m['final_training_days']['mean']
            fe_same = frontier_energy(f['flow_weighted_nh4'])
            tr = m['training'][-10:]
            md.append(f"| {NAMES[meth]} | {m['seed']} | {f['aeration_kwh_per_day']:.1f} | "
                      f"{100 * (f['aeration_kwh_per_day'] / dopi['aeration_kwh_per_day'] - 1):+.2f}% | "
                      f"{f['flow_weighted_nh4']:.3f} | {fe_same:.1f} | "
                      f"{100 * (f['aeration_kwh_per_day'] / fe_same - 1):+.2f}% | {f['nh4_violation_percent']:.2f}% | "
                      f"{m['training'][-1]['lambda']:.3f} | {100 * np.median([r['policy_clip_fraction'] for r in tr]):.1f} | "
                      f"{100 * np.median([r['value_clip_fraction'] for r in tr]):.1f} |")
    md += ['', '경계 대비: 같은 NH4 를 내는 고정 설정값(보간)의 에너지와 비교. 음수면 고정 설정값 경계보다 효율적이다.', '']
    md += ['## 14일 연속 운전 평가 (후반 7일)', '',
           '| 제어 | 날씨 | 폭기 kWh/일 | 유량 가중 NH4 | NH4 > 4 시간 | 일별 NH4 CVaR20 | 평균 설정값 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for w in ('dry', 'rain'):
        for name in ('fixed84', 'fixed240', 'do_pi', 'abac'):
            if name in refs:
                x = refs[name][w]['week']
                md.append(f"| {name} | {w} | {x['aeration_kwh_per_day']:.1f} | {x['flow_weighted_nh4']:.3f} | "
                          f"{x['nh4_violation_percent']:.2f}% | {x['daily_nh4_cvar20']:.3f} | {x.get('setpoint_mean', float('nan')):.2f} |")
        for meth in ('ppo', 'fcsrl'):
            for m in sorted(runs.get(meth, []), key=lambda m: m['seed']):
                x = m['final'][w]['week']
                md.append(f"| {NAMES[meth]} s{m['seed']} | {w} | {x['aeration_kwh_per_day']:.1f} | {x['flow_weighted_nh4']:.3f} | "
                          f"{x['nh4_violation_percent']:.2f}% | {x['daily_nh4_cvar20']:.3f} | {x['setpoint_mean']:.2f} |")
    (a.out / 'results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))

    (a.out / 'figures').mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(fe, fn, 'k.-', lw=1, label='Constant setpoint (0.75 to 3.0)')
    axs[0].scatter([dopi['aeration_kwh_per_day']], [lim_nh4], marker='*', s=150, c='k', label='DO PI (sp 2.0), start')
    if meta.get('abac_selected'):
        g = meta['abac_selected']
        axs[0].scatter([g['aeration_kwh_per_day']], [g['flow_weighted_nh4']], marker='s', c='grey', label='ABAC (tuned)')
    for meth in ('ppo', 'fcsrl'):
        for k, m in enumerate(sorted(runs.get(meth, []), key=lambda m: m['seed'])):
            f = m['final_training_days']['mean']
            axs[0].scatter([f['aeration_kwh_per_day']], [f['flow_weighted_nh4']], c=COLORS[meth],
                           label=NAMES[meth] if k == 0 else None)
            tr = m['training']
            u = [r['update'] for r in tr]
            axs[1].plot(u, [r['daily_aeration_kwh'] for r in tr], c=COLORS[meth], alpha=.7, lw=1)
            axs[2].plot(u, [r['lambda'] for r in tr], c=COLORS[meth], alpha=.7, lw=1, label=NAMES[meth] if k == 0 else None)
    axs[0].axhline(lim_nh4, ls='--', c='k', lw=.7)
    axs[0].set_xlabel('Aeration energy (kWh/day)')
    axs[0].set_ylabel('Flow-weighted NH4 (mg N/L)')
    axs[0].set_title('Training days, final policy', fontsize=9)
    axs[0].legend(fontsize=7)
    axs[1].axhline(dopi['aeration_kwh_per_day'], ls='--', c='k', lw=.7)
    axs[1].set_xlabel('Update')
    axs[1].set_ylabel('Training rollout aeration (kWh/day, with exploration)')
    axs[2].set_xlabel('Update')
    axs[2].set_ylabel('Lagrange multiplier')
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / 'figures' / 'e18_summary.png', dpi=180)


if __name__ == '__main__':
    main()
