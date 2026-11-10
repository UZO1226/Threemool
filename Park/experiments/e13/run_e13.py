"""Park/e13 차단 개입에 학습 비용을 준 정책 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e13.py one --cond A --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_intcost_runs --out <결과>/e13
    python run_e13.py report --out <결과>/e13 --runs <결과>/dw_intcost_runs --e12 <결과>/e12 --md results.md
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
from run_e07 import ms  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

TARGETS = (2.0, 2.5, 3.0, 3.5, 4.0)
CONDS = ('A', 'B')
E08_THETA3 = (91279, 6.62)  # e08 정책 + NH4 규칙 theta 3.0 의 seed 평균 (README 에 미리 적은 값)


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


def load(folder, cond):
    return {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(folder).glob(f'{cond}_seed_*.json'))}


def judge(new, old):
    """seed 별로 new < old 인지, 그리고 seed 평균 비교. old 가 없는 seed 는 성립하지 않은 것으로 센다."""
    ok = [s in old and new[s] < old[s] for s in new]
    mean_new, mean_old = float(np.mean(list(new.values()))), float(np.mean(list(old.values())))
    return sum(ok), mean_new, mean_old, sum(ok) >= 3 and mean_new < mean_old


def report(a):
    res = {c: load(a.out, c) for c in CONDS}
    e12 = load(a.e12, 'A')
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
            '| --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    L = ['FCSRL + PPO-Lag 정책 seed 5개씩, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', '',
         '## 학습 마지막 갱신', '',
         '| 조건 | seed | 라그랑주 승수 | 학습 하루 비용 (개입 비용 포함) | 실제 하루 비용 | 개입 % |', '| --- | --- | --- | --- | --- | --- |']
    for c in CONDS:
        for s in sorted(res[c]):
            t = json.loads((Path(a.runs) / f'{c}_s{s}' / 'training.json').read_text())[-1]
            L.append(f"| {c} | {s} | {t['lambda']:.3f} | {t['daily_cost_mean']:.2f} | {t['env_daily_cost_mean']:.2f} | {t['intervention_percent']:.1f} |")
    L += ['', '## 평가 (정책 평균)', ''] + head
    groups = [('e12 A', [e12[s] for s in sorted(e12)])] + [(c, [res[c][s] for s in sorted(res[c])]) for c in CONDS]
    for name, rs in groups:
        if not rs:
            continue
        L.append(row(f'{name} 차단 없음', [r['none'] for r in rs]))
        L += [row(f'{name} + 최소 개입 target {t}', [r['min1'][str(t)] for r in rs]) for t in TARGETS]
    L += ['', '## seed 평균 비용 <= 12 (PROTOCOL v1)', '']
    for c in CONDS:
        for s in sorted(res[c]):
            r = res[c][s]
            L.append(f"- {c} seed {s}: 차단 없음 {r['none']['cost']:.2f}, target 4.0 {r['min1']['4.0']['cost']:.2f},"
                     f" target 3.5 {r['min1']['3.5']['cost']:.2f} ({'만족' if r['min1']['3.5']['cost'] <= 12 else '불만족'})")
    old_none = {s: e12[s]['none']['cost'] for s in e12}
    old_40 = {s: e12[s]['min1']['4.0']['cost'] for s in e12}
    for c in CONDS:
        seeds = sorted(res[c])
        if not seeds:
            continue
        L += ['', f'## 조건 {c} 판정용 seed 별 값', '',
              '| seed | 차단 없음 비용 (e12 A) | target 4.0 비용 (e12 A) | target 3.5 공기량 | target 3.5 비용 | H3 성립 |',
              '| --- | --- | --- | --- | --- | --- |']
        h3, airs, costs = [], [], []
        for s in seeds:
            r = res[c][s]
            air, cost = r['min1']['3.5']['air_m3'], r['min1']['3.5']['cost']
            airs.append(air)
            costs.append(cost)
            h3.append(air < E08_THETA3[0] and cost < E08_THETA3[1])
            L.append(f"| {s} | {r['none']['cost']:.2f} ({old_none.get(s, float('nan')):.2f}) | "
                     f"{r['min1']['4.0']['cost']:.2f} ({old_40.get(s, float('nan')):.2f}) | {air:.0f} | {cost:.2f} | {'예' if h3[-1] else '아니오'} |")
        n1, m1, o1, ok1 = judge({s: res[c][s]['none']['cost'] for s in seeds}, old_none)
        n2, m2, o2, ok2 = judge({s: res[c][s]['min1']['4.0']['cost'] for s in seeds}, old_40)
        ma, mc = float(np.mean(airs)), float(np.mean(costs))
        ok3 = sum(h3) >= 3 and ma < E08_THETA3[0] and mc < E08_THETA3[1]
        L += ['', f'### 조건 {c} 사전 등록 판정', '',
              f"- H1 (차단 없음 비용 < 같은 seed e12 A): 평균 {m1:.2f} 대 {o1:.2f}, 성립 seed {n1}/{len(seeds)}: {'지지' if ok1 else '지지 안 됨'}",
              f"- H2 (target 4.0 비용 < 같은 seed e12 A): 평균 {m2:.2f} 대 {o2:.2f}, 성립 seed {n2}/{len(seeds)}: {'지지' if ok2 else '지지 안 됨'}",
              f"- H3 (target 3.5 에서 공기량 < 91,279 이고 비용 < 6.62): 평균 {ma:.0f}, {mc:.2f}, 성립 seed {sum(h3)}/{len(seeds)}: {'지지' if ok3 else '지지 안 됨'}"]
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
    for k in ('out', 'runs', 'e12', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
