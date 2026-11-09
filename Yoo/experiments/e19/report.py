"""e19 결과 표, 사전 기준 판정, 그림

    python report.py --prepared <준비 폴더> --main <결과 폴더> --out .

고정 설정값 경계를 주 평가와 같은 14일 연속 운전으로 재고 <준비 폴더>/eval_frontier.json 에 저장한다(처음 한 번).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e18'))
from setpoint_env import evaluate, setpoint_to_action  # noqa: E402

SPS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
NAMES = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO-Lag'}
COLORS = {'ppo': '#ff7f0e', 'fcsrl': '#2ca02c'}
KEYS = ('aeration_kwh_per_day', 'flow_weighted_nh4', 'nh4_violation_percent', 'daily_nh4_cvar20')


def _eval_sp(args):
    prepared, tanks, sp = args
    initial = dict(np.load(Path(prepared) / 'initial.npz'))
    act = setpoint_to_action(sp)
    r = evaluate(initial, lambda o, e: [act], tanks=tanks)
    return str(sp), {w: r[w]['week'] for w in r}


def eval_frontier(prepared, tanks):
    path = prepared / 'eval_frontier.json'
    if not path.exists():
        with Pool(len(SPS)) as p:
            res = dict(p.map(_eval_sp, [(str(prepared), tanks, sp) for sp in SPS]))
        path.write_text(json.dumps(res, indent=1))
    return json.loads(path.read_text())


def rel_energy(e, n, fe, fn):
    """같은 NH4 를 내는 고정 설정값 경계의 에너지(보간) 대비 % 차이. 경계 범위 밖이면 끝점 기울기로 외삽"""
    order = np.argsort(fn)
    fn, fe = np.asarray(fn)[order], np.asarray(fe)[order]
    if n < fn[0]:
        ref = fe[0] + (n - fn[0]) * (fe[1] - fe[0]) / (fn[1] - fn[0])
    elif n > fn[-1]:
        ref = fe[-1] + (n - fn[-1]) * (fe[-1] - fe[-2]) / (fn[-1] - fn[-2])
    else:
        ref = float(np.interp(n, fn, fe))
    return 100 * (e / ref - 1)


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float('nan'), float('nan')
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.prepared / 'metadata.json').read_text())
    refs = json.loads((a.prepared / 'references.json').read_text())
    tanks = tuple(meta['tanks'])
    fr = eval_frontier(a.prepared, tanks)
    runs = {}
    for f in sorted(a.main.glob('*/seed_*/metrics.json')):
        m = json.loads(f.read_text())
        runs.setdefault(m['method'], []).append(m)
    md = [f"제어 구조: {', '.join(map(str, tanks))}번 조 DO PI. 비용 한도 {meta['cost_limit']:.3f} "
          f"(학습 날짜 설정값 2.0 의 평균 NH4 {meta['do_pi_training_days']['mean']['flow_weighted_nh4']:.3f}).", '']
    verdict = {}
    for w in ('dry', 'rain'):
        fe = [fr[str(s)][w]['aeration_kwh_per_day'] for s in SPS]
        fn = [fr[str(s)][w]['flow_weighted_nh4'] for s in SPS]
        md += [f'## 14일 연속 운전 후반 7일, {"건기 (주 평가)" if w == "dry" else "우기 (보조)"}', '',
               '| 제어 | seed | 폭기 kWh/일 | 유량 가중 NH4 | NH4 > 4 시간 | 일별 NH4 CVaR20 | 평균 설정값 | 경계 대비 에너지 |',
               '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for s in SPS:
            x = fr[str(s)][w]
            md.append(f"| 고정 설정값 {s} | | {x['aeration_kwh_per_day']:.1f} | {x['flow_weighted_nh4']:.3f} | "
                      f"{x['nh4_violation_percent']:.2f}% | {x['daily_nh4_cvar20']:.3f} | {s} | 0 (경계) |")
        for name in ('abac', 'feedforward'):
            if name in refs:
                x = refs[name][w]['week']
                md.append(f"| {name} | | {x['aeration_kwh_per_day']:.1f} | {x['flow_weighted_nh4']:.3f} | "
                          f"{x['nh4_violation_percent']:.2f}% | {x['daily_nh4_cvar20']:.3f} | {x['setpoint_mean']:.2f} | "
                          f"{rel_energy(x['aeration_kwh_per_day'], x['flow_weighted_nh4'], fe, fn):+.2f}% |")
        summ = {}
        for meth in ('ppo', 'fcsrl'):
            rows = []
            for m in sorted(runs.get(meth, []), key=lambda m: m['seed']):
                x = m['final'][w]['week']
                r = rel_energy(x['aeration_kwh_per_day'], x['flow_weighted_nh4'], fe, fn)
                rows.append((x, r))
                md.append(f"| {NAMES[meth]} | {m['seed']} | {x['aeration_kwh_per_day']:.1f} | {x['flow_weighted_nh4']:.3f} | "
                          f"{x['nh4_violation_percent']:.2f}% | {x['daily_nh4_cvar20']:.3f} | {x['setpoint_mean']:.2f} | {r:+.2f}% |")
            if rows:
                summ[meth] = {k: [x[k] for x, _ in rows] for k in KEYS}
                summ[meth]['rel'] = [r for _, r in rows]
        md.append('')
        dopi = fr['2.0'][w]
        md += ['| 방법 | seed 수 | 폭기 kWh/일 | NH4 | DO PI 대비 NH4 | 경계 대비 에너지 (95% CI) |', '| --- | --- | --- | --- | --- | --- |']
        for meth, sm in summ.items():
            lo, hi = ci95(sm['rel'])
            md.append(f"| {NAMES[meth]} | {len(sm['rel'])} | {np.mean(sm['aeration_kwh_per_day']):.1f} ± {np.std(sm['aeration_kwh_per_day'], ddof=1) if len(sm['rel']) > 1 else 0:.1f} | "
                      f"{np.mean(sm['flow_weighted_nh4']):.3f} ± {np.std(sm['flow_weighted_nh4'], ddof=1) if len(sm['rel']) > 1 else 0:.3f} | "
                      f"{np.mean(sm['flow_weighted_nh4']) / dopi['flow_weighted_nh4']:.3f}배 | "
                      f"{np.mean(sm['rel']):+.2f}% ({lo:+.2f}, {hi:+.2f}) |")
            if w == 'dry':
                ff = refs.get('feedforward', {}).get(w, {}).get('week')
                ff_rel = rel_energy(ff['aeration_kwh_per_day'], ff['flow_weighted_nh4'], fe, fn) if ff else float('nan')
                verdict[meth] = {
                    '1_수질_유지': bool(np.mean(sm['flow_weighted_nh4']) <= 1.05 * dopi['flow_weighted_nh4']),
                    '2_경계_개선': bool(np.mean(sm['rel']) < 0 and hi < 0),
                    '3_규칙_대비': bool(np.mean(sm['rel']) < ff_rel), 'feedforward_rel': ff_rel,
                    'rel_mean': float(np.mean(sm['rel'])), 'rel_ci95': [float(lo), float(hi)],
                    'nh4_ratio': float(np.mean(sm['flow_weighted_nh4']) / dopi['flow_weighted_nh4'])}
        if len(summ) == 2 and all(len(v['rel']) > 1 for v in summ.values()):
            tr = stats.ttest_ind(summ['fcsrl']['rel'], summ['ppo']['rel'], equal_var=False).pvalue
            tn = stats.ttest_ind(summ['fcsrl']['flow_weighted_nh4'], summ['ppo']['flow_weighted_nh4'], equal_var=False).pvalue
            md += ['', f'FCSRL + PPO-Lag 대 PPO-Lag Welch t 검정: 경계 대비 에너지 p = {tr:.3f}, NH4 p = {tn:.3f}', '']
            if w == 'dry':
                verdict['4_fcsrl_대_ppo'] = {'rel_p': float(tr), 'nh4_p': float(tn)}
        md.append('')
    md += ['## 사전 기준 판정 (건기 주 평가)', '', '```', json.dumps(verdict, indent=1, ensure_ascii=False), '```', '']
    (a.out / 'results.md').write_text('\n'.join(md))
    (a.out / 'verdict.json').write_text(json.dumps(verdict, indent=1, ensure_ascii=False))
    print('\n'.join(md))

    (a.out / 'figures').mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, w in zip(axs, ('dry', 'rain')):
        fe = [fr[str(s)][w]['aeration_kwh_per_day'] for s in SPS]
        fn = [fr[str(s)][w]['flow_weighted_nh4'] for s in SPS]
        ax.plot(fe, fn, 'k.-', lw=1, label='Constant setpoint')
        ax.scatter([fr['2.0'][w]['aeration_kwh_per_day']], [fr['2.0'][w]['flow_weighted_nh4']], marker='*', s=150, c='k',
                   label='DO PI sp 2.0 (start)', zorder=3)
        for name, mk in (('feedforward', 'D'), ('abac', 's')):
            if name in refs:
                x = refs[name][w]['week']
                ax.scatter([x['aeration_kwh_per_day']], [x['flow_weighted_nh4']], marker=mk, c='grey', label=name)
        for meth in ('ppo', 'fcsrl'):
            ms = runs.get(meth, [])
            ax.scatter([m['final'][w]['week']['aeration_kwh_per_day'] for m in ms],
                       [m['final'][w]['week']['flow_weighted_nh4'] for m in ms], c=COLORS[meth], label=NAMES[meth], alpha=.8)
        ax.set_title('Dry (main evaluation)' if w == 'dry' else 'Rain (not used in training)', fontsize=9)
        ax.set_xlabel('Aeration energy (kWh/day)')
        ax.set_ylabel('Flow-weighted NH4 (mg N/L)')
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / 'figures' / 'e19_frontier.png', dpi=180)


if __name__ == '__main__':
    main()
