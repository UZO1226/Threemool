"""Park/e12 학습 중에도 최소 개입 차단을 켠 정책 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e12.py one --cond A --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_minbrake_runs --out <결과>/e12
    python run_e12.py report --out <결과>/e12 --runs <결과>/dw_minbrake_runs --e10 <결과>/e10 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e07'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from run_e07 import gaps, ms  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

TARGETS = (2.0, 2.5, 3.0, 3.5, 4.0)
CONDS = ('A', 'B')
E08_THETA3 = (91279, 6.62)    # e08 정책 + NH4 규칙 theta 3.0 의 seed 평균 (README 에 미리 적은 값)
E10_TARGET35 = (90747, 5.46)  # e08 정책 + 최소 개입 1스텝 target 3.5 의 seed 평균


def run_one(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    agent = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    agent.load(Path(a.runs) / f'{a.cond}_s{a.seed}' / 'model_final')
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    out = {'cond': a.cond, 'seed': a.seed, 'none': evaluate(env, splits['val'], act),
           'min1': {str(t): evaluate(env, splits['val'], act, shield=MinAirBrake(dyn, t, 1, 'others')) for t in TARGETS}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.cond}_seed_{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.cond, a.seed, flush=True)


def report(a):
    res = {}
    for p in Path(a.out).glob('*_seed_*.json'):
        r = json.loads(p.read_text())
        res.setdefault(r['cond'], {})[r['seed']] = r
    e10 = {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(a.e10).glob('seed_*.json'))}
    keys = ('air_m3', 'cost')
    ref = {'none': {k: float(np.mean([e10[s]['none'][k] for s in e10])) for k in keys}}
    for t in TARGETS:
        ref[str(t)] = {k: float(np.mean([e10[s]['min1'][str(t)][k] for s in e10])) for k in keys}
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
            '| --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    L = ['FCSRL + PPO-Lag 정책 seed 5개씩, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', '',
         '## 학습 마지막 갱신', '', '| 조건 | seed | 라그랑주 승수 | 학습 하루 비용 | 개입 % |', '| --- | --- | --- | --- | --- |']
    for c in CONDS:
        for s in sorted(res.get(c, {})):
            t = json.loads((Path(a.runs) / f'{c}_s{s}' / 'training.json').read_text())[-1]
            L.append(f"| {c} | {s} | {t['lambda']:.3f} | {t['daily_cost_mean']:.2f} | {t['intervention_percent']:.1f} |")
    L += ['', '## 평가 (정책 평균)', ''] + head
    L.append(row('기준: e08 정책 차단 없음', [e10[s]['none'] for s in sorted(e10)]))
    L += [row(f'기준: e08 정책 + 최소 개입 target {t}', [e10[s]['min1'][str(t)] for s in sorted(e10)]) for t in TARGETS]
    for c in CONDS:
        rs = [res[c][s] for s in sorted(res.get(c, {}))]
        if not rs:
            continue
        L.append(row(f'{c} 차단 없음', [r['none'] for r in rs]))
        L += [row(f'{c} + 최소 개입 target {t}', [r['min1'][str(t)] for r in rs]) for t in TARGETS]
    L += ['', '## seed 평균 비용 <= 12 (PROTOCOL v1)', '']
    for c in CONDS:
        for s in sorted(res.get(c, {})):
            r = res[c][s]
            L.append(f"- {c} seed {s}: 차단 없음 {r['none']['cost']:.2f}, target 3.5 {r['min1']['3.5']['cost']:.2f}"
                     f" ({'만족' if r['min1']['3.5']['cost'] <= 12 else '불만족'})")
    for c in CONDS:
        seeds = sorted(res.get(c, {}))
        if not seeds:
            continue
        L += ['', f'## 조건 {c} 판정용 seed 별 값', '',
              '| seed | H1 같은 공기량 차이 (점) | target 3.5 공기량 | target 3.5 하루 비용 | H2 성립 | H3 성립 |',
              '| --- | --- | --- | --- | --- | --- |']
        h1, h2, h3, airs, costs = [], [], [], [], []
        for s in seeds:
            r = res[c][s]
            g = gaps(r['min1'], ref)
            h1.append(float(np.mean(g)) if g else None)
            air, cost = r['min1']['3.5']['air_m3'], r['min1']['3.5']['cost']
            airs.append(air)
            costs.append(cost)
            h2.append(air < E08_THETA3[0] and cost < E08_THETA3[1])
            h3.append(air < E10_TARGET35[0] and cost < E10_TARGET35[1])
            h1_txt = '없음 (0)' if not g else f'{np.mean(g):.2f} ({len(g)})'
            L.append(f"| {s} | {h1_txt} | {air:.0f} | {cost:.2f} | {'예' if h2[-1] else '아니오'} | {'예' if h3[-1] else '아니오'} |")
        known = [v for v in h1 if v is not None]
        n1 = sum(v is not None and v < 0 for v in h1)
        ok1 = n1 >= 3 and bool(known) and np.mean(known) < 0
        ma, mc = float(np.mean(airs)), float(np.mean(costs))
        ok2 = sum(h2) >= 3 and ma < E08_THETA3[0] and mc < E08_THETA3[1]
        ok3 = sum(h3) >= 3 and ma < E10_TARGET35[0] and mc < E10_TARGET35[1]
        L += ['', f'### 조건 {c} 사전 등록 판정', '',
              f"- H1 (같은 공기량 차이 < 0): 평균 {np.mean(known) if known else float('nan'):.2f}, 성립 seed {n1}/{len(seeds)}: {'지지' if ok1 else '지지 안 됨'}",
              f"- H2 (target 3.5 에서 공기량 < 91,279 이고 비용 < 6.62): 평균 {ma:.0f}, {mc:.2f}, 성립 seed {sum(h2)}/{len(seeds)}: {'지지' if ok2 else '지지 안 됨'}",
              f"- H3 (target 3.5 에서 공기량 < 90,747 이고 비용 < 5.46): 평균 {ma:.0f}, {mc:.2f}, 성립 seed {sum(h3)}/{len(seeds)}: {'지지' if ok3 else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--cond', choices=CONDS, required=True)
    o.add_argument('--seed', type=int, required=True)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'runs', 'e10', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
