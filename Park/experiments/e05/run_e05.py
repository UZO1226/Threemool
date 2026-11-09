"""Park/e05 위험에 비례해 미리 폭기를 올리는 차단 (Park 작성). 규칙은 README.md 를 따른다.

    export BSM1_ARCHIVE=<BSM1.zip> WANDB_MODE=disabled
    python run_e05.py one --gamma 0.9 --seed 0 --prepared <결과>/prepared --runs <결과>/runs --out <결과>/e05
    python run_e05.py report --out <결과>/e05 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
for name in ('e01', 'e02', 'e03', 'e04'):
    sys.path.insert(0, str(HERE.parent / name))
from agent import FCSRL, Config, ObsNormalizer  # noqa: E402
from evaluate import run  # noqa: E402
from shield import GradedShield, NH4Shield, PredictorShield  # noqa: E402
from run_e02 import model_dir  # noqa: E402
from run_e03 import Logistic, features  # noqa: E402
from run_e04 import gaps, train_samples  # noqa: E402

GAMMAS = (0.9, 0.97, 0.99)
CURVES = {
    'graded': (0.1, 0.25, 0.5, 1.0, 2.0),
    'onoff': (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97),
    'nh4_rule': (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
}
LABELS = {'graded': '비례 하한 c', 'onoff': '켜고 끄기 기준', 'nh4_rule': 'NH4 theta'}
COMPARISONS = (('H1', 'graded', 'onoff'), ('H2', 'graded', 'nh4_rule'), ('H3', 'onoff', 'nh4_rule'))


def run_one(a):
    prepared = Path(a.prepared)
    agent = FCSRL(11, 1, Config(feasibility_discount=a.gamma),
                  ObsNormalizer(np.load(prepared / 'train_observations.npy')))
    agent.load(model_dir(a.runs, a.gamma, a.seed))
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(7)]
    initial = dict(np.load(prepared / 'initial.npz'))
    x, y = train_samples(agent, states)
    model = Logistic().fit(features(x, 'all_obs'), y)

    def prob(obs):
        z = model.score(features(np.atleast_2d(obs), 'all_obs'))[0]
        return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))

    makers = {'graded': lambda v: GradedShield(prob, v), 'onoff': lambda v: PredictorShield(prob, v),
              'nh4_rule': lambda v: NH4Shield(v)}
    out = {'gamma': a.gamma, 'seed': a.seed}
    w = run(initial, agent.act_det, 'dry')['week']
    out['none'] = {'kwh': w['aeration_kwh_per_day'], 'violation': w['step_violation_percent'], 'intervention': 0.0}
    for kind, values in CURVES.items():
        out[kind] = {}
        for v in values:
            w = run(initial, agent.act_det, 'dry', shield=makers[kind](v))['week']
            out[kind][str(v)] = {'kwh': w['aeration_kwh_per_day'], 'violation': w['step_violation_percent'],
                                 'intervention': w['intervention_percent']}
        print(a.gamma, a.seed, kind, 'done', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'g{a.gamma}_s{a.seed}.json').write_text(json.dumps(out, indent=2))


def ms(v, d=2):
    v = np.asarray(v, float)
    return f'{v.mean():.{d}f} ± {v.std():.{d}f}'


def report(a):
    res = {}
    for p in Path(a.out).glob('g*_s*.json'):
        r = json.loads(p.read_text())
        res[(r['gamma'], r['seed'])] = r
    keys = sorted(res)
    L = [f'정책 {len(keys)}개 (gamma_F x seed), 평균 ± 표준편차(ddof=0). 건기 평가 구간 (14일 운전의 후반 7일).', '',
         '## 곡선 (정책 평균)', '', '| 조건 | kWh/day | 순간 위반 % | 개입 % |', '| --- | --- | --- | --- |']
    rs = [res[k] for k in keys]
    L.append(f"| 차단 없음 | {ms([r['none']['kwh'] for r in rs], 0)} | {ms([r['none']['violation'] for r in rs])} | 0 |")
    for kind, values in CURVES.items():
        for v in values:
            ps = [r[kind][str(v)] for r in rs]
            L.append(f"| {LABELS[kind]} {v} | {ms([p['kwh'] for p in ps], 0)} | {ms([p['violation'] for p in ps])} | "
                     f"{ms([p['intervention'] for p in ps], 1)} |")
    L += ['', '## 같은 에너지 차이 (앞 곡선 위반 % 빼기 비교 곡선 보간 위반 %, 음수면 앞 곡선이 낫다)', '',
          '| gamma_F | seed | 비례 대 켜고 끄기 (점) | 비례 대 NH4 (점) | 켜고 끄기 대 NH4 (점) |', '| --- | --- | --- | --- | --- |']
    table = {}
    for k in keys:
        row = []
        for hid, front, ref in COMPARISONS:
            g = gaps(res[k][front], res[k][ref])
            table[(hid, k)] = float(np.mean(g)) if g else None
            row.append(f"{'없음' if not g else f'{np.mean(g):.2f}'} ({len(g)})")
        L.append(f'| {k[0]} | {k[1]} | ' + ' | '.join(row) + ' |')
    L += ['', '## 사전 등록 판정 (정책 9개 중 6개 이상 음수, 평균 음수)', '']
    for hid, front, ref in COMPARISONS:
        vals = [table[(hid, k)] for k in keys]
        known = [v for v in vals if v is not None]
        neg = sum(v is not None and v < 0 for v in vals)
        ok = neg >= 6 and known and np.mean(known) < 0
        mean_txt = f'{np.mean(known):.2f}' if known else '없음'
        L.append(f'- {hid} ({LABELS[front].split()[0]} 대 {LABELS[ref].split()[0]}): 평균 {mean_txt} %p, '
                 f'음수 {neg}/{len(keys)}, 비교 불가 {len(vals) - len(known)}: {"지지" if ok else "지지 안 됨"}')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    for k in ('prepared', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    o.add_argument('--gamma', type=float, required=True)
    o.add_argument('--seed', type=int, required=True)
    r = sub.add_parser('report')
    r.add_argument('--out', required=True)
    r.add_argument('--md', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
