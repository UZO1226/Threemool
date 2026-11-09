"""Park/e02 조기 경보 AUROC 와 같은 에너지 비교 (Park 작성). 규칙은 README.md 를 따른다.

    export BSM1_ARCHIVE=<BSM1.zip> WANDB_MODE=disabled
    python run_e02.py one --gamma 0.97 --seed 0 --prepared <결과>/prepared --runs <결과>/runs --out <결과>/e02
    python run_e02.py report --out <결과>/e02 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e01'))
from agent import FCSRL, Config, ObsNormalizer  # noqa: E402
from evaluate import run  # noqa: E402
from shield import FeasibilityShield, NH4Shield  # noqa: E402
from run_e01 import auroc, train_days  # noqa: E402

GAMMAS = (0.9, 0.97, 0.99)
HORIZONS = (12, 24)
TAUS = (0.3, 0.5, 0.7, 0.9)
THETAS = (3.0, 3.5, 4.0, 4.5)


def model_dir(runs, gamma, seed):
    name = f'fcsrl_s{seed}' if gamma == 0.9 else f'fcsrl_g{gamma}_s{seed}'
    return Path(runs) / name / 'model_final'


def early_warning(days, horizon):
    f, n, y = [], [], []
    for d in days:
        v = d['violation']
        for t in range(len(v) - horizon):
            if v[t]:
                continue
            y.append(v[t + 1:t + horizon + 1].any())
            f.append(d['f'][t])
            n.append(d['nh4_obs'][t])
    return {'n_steps': len(y), 'positive_rate': float(np.mean(y)),
            'auroc_feasibility': auroc(f, y), 'auroc_nh4_obs': auroc(n, y)}


def point(week):
    return {'kwh': week['aeration_kwh_per_day'], 'violation': week['step_violation_percent'],
            'intervention': week.get('intervention_percent', 0.0)}


def run_one(a):
    prepared = Path(a.prepared)
    normalizer = ObsNormalizer(np.load(prepared / 'train_observations.npy'))
    agent = FCSRL(11, 1, Config(feasibility_discount=a.gamma), normalizer)
    agent.load(model_dir(a.runs, a.gamma, a.seed))
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(7)]
    initial = dict(np.load(prepared / 'initial.npz'))

    days = train_days(agent, states, scorer=FeasibilityShield(agent, 1.0))
    out = {'gamma': a.gamma, 'seed': a.seed,
           'early_warning': {str(h): early_warning(days, h) for h in HORIZONS}}
    pol = agent.act_det
    out['none'] = point(run(initial, pol, 'dry')['week'])
    out['feasibility'] = {str(t): point(run(initial, pol, 'dry', shield=FeasibilityShield(agent, t))['week']) for t in TAUS}
    out['nh4_rule'] = {str(t): point(run(initial, pol, 'dry', shield=NH4Shield(t))['week']) for t in THETAS}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'g{a.gamma}_s{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.gamma, a.seed, flush=True)


def matched_gaps(r):
    """F 차단 각 점과 같은 에너지에서 NH4 규칙 곡선(선형 보간) 의 위반 차이."""
    rule = sorted(r['nh4_rule'].values(), key=lambda p: p['kwh'])
    xs, ys = [p['kwh'] for p in rule], [p['violation'] for p in rule]
    gaps = []
    for tau, p in r['feasibility'].items():
        if xs[0] <= p['kwh'] <= xs[-1]:
            gaps.append({'tau': tau, 'gap': p['violation'] - float(np.interp(p['kwh'], xs, ys))})
    return gaps


def ms(v, d=3):
    v = np.asarray(v, float)
    return f'{v.mean():.{d}f} ± {v.std():.{d}f}'


def report(a):
    res = {}
    for p in Path(a.out).glob('g*_s*.json'):
        r = json.loads(p.read_text())
        res[(r['gamma'], r['seed'])] = r
    seeds = sorted({s for _, s in res})
    L = [f'seed {len(seeds)}개, 평균 ± 표준편차(ddof=0). 건기, 학습 날짜(A0) 와 평가 구간(A1).', '']
    for h in HORIZONS:
        L += [f'## 조기 경보 AUROC, H = {h} ({h * 5}분), 현재 위반이 아닌 스텝만', '',
              '| gamma_F | 양성 비율 | AUROC F | AUROC NH4 측정값 | F 가 높은 seed |', '| --- | --- | --- | --- | --- |']
        for g in GAMMAS:
            rs = [res[(g, s)]['early_warning'][str(h)] for s in seeds if (g, s) in res]
            wins = sum(r['auroc_feasibility'] > r['auroc_nh4_obs'] for r in rs)
            L.append(f"| {g} | {ms([r['positive_rate'] for r in rs])} | {ms([r['auroc_feasibility'] for r in rs])} | "
                     f"{ms([r['auroc_nh4_obs'] for r in rs])} | {wins}/{len(rs)} |")
        L.append('')
    L += ['## seed 별 조기 경보 AUROC (H = 12)', '', '| gamma_F | seed | F | NH4 측정값 |', '| --- | --- | --- | --- |']
    for g in GAMMAS:
        for s in seeds:
            if (g, s) in res:
                r = res[(g, s)]['early_warning']['12']
                L.append(f"| {g} | {s} | {r['auroc_feasibility']:.3f} | {r['auroc_nh4_obs']:.3f} |")
    L += ['', '## 평가 구간 에너지 대 순간 위반 (seed 평균)', '',
          '| gamma_F | 조건 | kWh/day | 순간 위반 % | 개입 % |', '| --- | --- | --- | --- | --- |']
    for g in GAMMAS:
        rs = [res[(g, s)] for s in seeds if (g, s) in res]
        L.append(f"| {g} | 차단 없음 | {ms([r['none']['kwh'] for r in rs], 0)} | {ms([r['none']['violation'] for r in rs], 2)} | 0 |")
        for kind, label, vals in (('feasibility', 'F tau', TAUS), ('nh4_rule', 'NH4 theta', THETAS)):
            for v in vals:
                ps = [r[kind][str(v)] for r in rs]
                L.append(f"| {g} | {label} {v} | {ms([p['kwh'] for p in ps], 0)} | {ms([p['violation'] for p in ps], 2)} | "
                         f"{ms([p['intervention'] for p in ps], 1)} |")
    L += ['', '## 같은 에너지에서 F 위반 % 와 NH4 규칙 위반 % 의 차이 (음수면 F 가 낫다)', '',
          '| gamma_F | seed | 비교 가능한 tau | 평균 차이 %p |', '| --- | --- | --- | --- |']
    gap_mean = {}
    for g in GAMMAS:
        gap_mean[g] = {}
        for s in seeds:
            if (g, s) not in res:
                continue
            gs = matched_gaps(res[(g, s)])
            m = float(np.mean([x['gap'] for x in gs])) if gs else None
            gap_mean[g][s] = m
            L.append(f"| {g} | {s} | {', '.join(x['tau'] for x in gs) or '없음'} | {'' if m is None else f'{m:.2f}'} |")

    def ew(g, key):
        return [res[(g, s)]['early_warning']['12'][key] for s in seeds if (g, s) in res]

    f_means = {g: np.mean(ew(g, 'auroc_feasibility')) for g in GAMMAS}
    h1 = f_means[0.9] < f_means[0.97] < f_means[0.99]
    h2, h3 = {}, {}
    for g in (0.97, 0.99):
        f, n = np.array(ew(g, 'auroc_feasibility')), np.array(ew(g, 'auroc_nh4_obs'))
        h2[g] = (f > n).sum() >= 2 and f.mean() > n.mean()
        gm = [v for v in gap_mean[g].values() if v is not None]
        h3[g] = len(gm) >= 2 and sum(v < 0 for v in gm) >= 2 and np.mean(gm) < 0
    L += ['', '## 사전 등록 판정', '',
          f"- H1 ({' < '.join(f'{f_means[g]:.3f}' for g in GAMMAS)}): {'지지' if h1 else '지지 안 됨'}",
          f"- H2: 0.97 {'지지' if h2[0.97] else '지지 안 됨'}, 0.99 {'지지' if h2[0.99] else '지지 안 됨'}",
          f"- H3: 0.97 {'지지' if h3[0.97] else '지지 안 됨'}, 0.99 {'지지' if h3[0.99] else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--gamma', type=float, required=True)
    o.add_argument('--seed', type=int, required=True)
    o.add_argument('--prepared', required=True)
    o.add_argument('--runs', required=True)
    o.add_argument('--out', required=True)
    r = sub.add_parser('report')
    r.add_argument('--out', required=True)
    r.add_argument('--md', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
