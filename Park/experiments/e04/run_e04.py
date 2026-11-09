"""Park/e04 단순 위반 예측기로 거는 안전 차단 (Park 작성). 규칙은 README.md 를 따른다.

    export BSM1_ARCHIVE=<BSM1.zip> WANDB_MODE=disabled
    python run_e04.py one --gamma 0.9 --seed 0 --prepared <결과>/prepared --runs <결과>/runs --e02 <결과>/e02 --out <결과>/e04
    python run_e04.py report --out <결과>/e04 --e02 <결과>/e02 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
for name in ('e01', 'e02', 'e03'):
    sys.path.insert(0, str(HERE.parent / name))
from agent import FCSRL, Config, ObsNormalizer  # noqa: E402
from env import AerationEnv  # noqa: E402
from evaluate import run  # noqa: E402
from shield import PredictorShield  # noqa: E402
from run_e02 import model_dir  # noqa: E402
from run_e03 import Logistic, features  # noqa: E402

GAMMAS = (0.9, 0.97, 0.99)
HORIZON = 12
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)


def train_samples(agent, states):
    """학습 날짜 궤적에서 (관측, 정답). 현재 위반 중인 스텝도 포함한다."""
    xs, ys = [], []
    for day, s in enumerate(states):
        env = AerationEnv(s, 'dry', start_day=day, days=1)
        obs, _ = env.reset()
        o, v = [], []
        while env.t < env.T:
            o.append(obs.copy())
            obs, _, _, _, info = env.step(np.asarray(agent.act_det(obs), float).reshape(1))
            v.append(info['violation'])
        v = np.array(v)
        for t in range(len(v) - HORIZON):
            xs.append(o[t])
            ys.append(v[t + 1:t + HORIZON + 1].any())
    return np.array(xs), np.array(ys, float)


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

    out = {'gamma': a.gamma, 'seed': a.seed, 'train_positive_rate': float(y.mean()), 'predictor': {}}
    for thr in THRESHOLDS:
        w = run(initial, agent.act_det, 'dry', shield=PredictorShield(prob, thr))['week']
        out['predictor'][str(thr)] = {'kwh': w['aeration_kwh_per_day'], 'violation': w['step_violation_percent'],
                                      'intervention': w['intervention_percent']}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'g{a.gamma}_s{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.gamma, a.seed, flush=True)


def gaps(points, curve):
    curve = sorted(curve.values(), key=lambda p: p['kwh'])
    xs, ys = [p['kwh'] for p in curve], [p['violation'] for p in curve]
    return [p['violation'] - float(np.interp(p['kwh'], xs, ys)) for p in points.values() if xs[0] <= p['kwh'] <= xs[-1]]


def ms(v, d=2):
    v = np.asarray(v, float)
    return f'{v.mean():.{d}f} ± {v.std():.{d}f}'


def report(a):
    res, base = {}, {}
    for p in Path(a.out).glob('g*_s*.json'):
        r = json.loads(p.read_text())
        res[(r['gamma'], r['seed'])] = r
        base[(r['gamma'], r['seed'])] = json.loads((Path(a.e02) / p.name).read_text())
    seeds = sorted({s for _, s in res})
    L = [f'seed {len(seeds)}개, 평균 ± 표준편차(ddof=0). 건기 평가 구간 (14일 운전의 후반 7일).', '',
         '## 예측기 차단 (관측 전체 로지스틱 회귀, 1시간 위반 확률)', '',
         '| gamma_F | 기준 확률 | kWh/day | 순간 위반 % | 개입 % |', '| --- | --- | --- | --- | --- |']
    for g in GAMMAS:
        rs = [res[(g, s)] for s in seeds if (g, s) in res]
        L.append(f"| {g} | 차단 없음 | {ms([base[(g, s)]['none']['kwh'] for s in seeds], 0)} | "
                 f"{ms([base[(g, s)]['none']['violation'] for s in seeds])} | 0 |")
        for t in THRESHOLDS:
            ps = [r['predictor'][str(t)] for r in rs]
            L.append(f"| {g} | {t} | {ms([p['kwh'] for p in ps], 0)} | {ms([p['violation'] for p in ps])} | "
                     f"{ms([p['intervention'] for p in ps], 1)} |")
    L += ['', '## 같은 에너지 차이 (예측기 차단 위반 % 빼기 비교 곡선 보간 위반 %, 음수면 예측기가 낫다)', '',
          '| gamma_F | seed | F 곡선 대비 | 비교 점 수 | NH4 규칙 곡선 대비 | 비교 점 수 |', '| --- | --- | --- | --- | --- | --- |']
    summary = {}
    for g in GAMMAS:
        for s in seeds:
            if (g, s) not in res:
                continue
            gf = gaps(res[(g, s)]['predictor'], base[(g, s)]['feasibility'])
            gn = gaps(res[(g, s)]['predictor'], base[(g, s)]['nh4_rule'])
            mf = float(np.mean(gf)) if gf else None
            mn = float(np.mean(gn)) if gn else None
            summary[(g, s)] = (mf, mn)
            L.append(f"| {g} | {s} | {'' if mf is None else f'{mf:.2f}'} | {len(gf)} | "
                     f"{'' if mn is None else f'{mn:.2f}'} | {len(gn)} |")
    L += ['', '## 사전 등록 판정 (gamma_F 0.9)', '']
    for hid, idx, label in (('H1', 0, 'F 곡선'), ('H2', 1, 'NH4 규칙 곡선')):
        vals = [summary.get((0.9, s), (None, None))[idx] for s in seeds]
        neg = sum(v is not None and v < 0 for v in vals)
        known = [v for v in vals if v is not None]
        ok = neg >= 2 and known and np.mean(known) < 0
        mean_txt = f'{np.mean(known):.2f}' if known else '없음'
        L.append(f'- {hid} ({label} 대비): 평균 {mean_txt} %p, 음수 seed {neg}/{len(seeds)}: {"지지" if ok else "지지 안 됨"}')
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
    o.add_argument('--e02')
    r = sub.add_parser('report')
    r.add_argument('--out', required=True)
    r.add_argument('--e02', required=True)
    r.add_argument('--md', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
