"""Park/e14 양방향 차단 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e14.py one --cond A --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_trim_runs --e12-runs <결과>/dw_minbrake_runs --out <결과>/e14
    python run_e14.py report --out <결과>/e14 --runs <결과>/dw_trim_runs --e12 <결과>/e12 --md results.md
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

CONDS = ('A', 'B', 'E')
MAIN = {'A': 'trim_3.5', 'B': 'trim_3.0', 'E': 'trim_3.5'}   # 각 조건의 대표값 (README 에 미리 적음)
SETTINGS = [('one_3.0', 3.0, False), ('one_3.5', 3.5, False), ('trim_3.0', 3.0, True), ('trim_3.5', 3.5, True), ('trim_4.0', 4.0, True)]


def run_one(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    agent = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    folder = Path(a.e12_runs) / f'A_s{a.seed}' if a.cond == 'E' else Path(a.runs) / f'{a.cond}_s{a.seed}'
    agent.load(folder / 'model_final')
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    out = {'cond': a.cond, 'seed': a.seed, 'none': evaluate(env, splits['val'], act)}
    for name, target, trim in SETTINGS:
        brake = MinAirBrake(dyn, target, 1, 'others', trim=trim)
        out[name] = evaluate(env, splits['val'], act, shield=brake)
        out[name]['lowered_pct'] = 100 * brake.lowered_steps / max(brake.steps, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.cond}_seed_{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.cond, a.seed, flush=True)


def load(folder, cond):
    return {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(folder).glob(f'{cond}_seed_*.json'))}


def report(a):
    res = {c: load(a.out, c) for c in CONDS}
    e12 = load(a.e12, 'A')
    base = {s: e12[s]['min1']['3.5'] for s in e12}
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % | 내린 스텝 % |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} | "
                f"{ms([p.get('lowered_pct', 0) for p in ps], 1)} |")
    L = ['FCSRL + PPO-Lag 정책 seed 5개씩, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.',
         'one 은 한 방향 최소 개입 차단 (e10), trim 은 양방향 차단 (e14). E 는 e12 A 정책이다.', '',
         '## 학습 마지막 갱신', '', '| 조건 | seed | 라그랑주 승수 | 학습 하루 비용 | 개입 % |', '| --- | --- | --- | --- | --- |']
    for c in ('A', 'B'):
        for s in sorted(res[c]):
            t = json.loads((Path(a.runs) / f'{c}_s{s}' / 'training.json').read_text())[-1]
            L.append(f"| {c} | {s} | {t['lambda']:.3f} | {t['daily_cost_mean']:.2f} | {t['intervention_percent']:.1f} |")
    L += ['', '## 평가 (정책 평균)', ''] + head
    L.append(row('기준: e12 A + one 3.5 (e12 결과 파일)', [base[s] for s in sorted(base)]))
    for c in CONDS:
        rs = [res[c][s] for s in sorted(res[c])]
        if not rs:
            continue
        L.append(row(f'{c} 차단 없음', [r['none'] for r in rs]))
        L += [row(f'{c} + {name}' + (' (대표)' if name == MAIN[c] else ''), [r[name] for r in rs]) for name, _, _ in SETTINGS]
    L += ['', '## seed 평균 비용 <= 12 (PROTOCOL v1, 대표값)', '']
    for c in CONDS:
        for s in sorted(res[c]):
            v = res[c][s][MAIN[c]]['cost']
            L.append(f"- {c} seed {s}: {v:.2f} ({'만족' if v <= 12 else '불만족'}), 차단 없음 {res[c][s]['none']['cost']:.2f}")
    for k, c in enumerate(CONDS, 1):
        seeds = sorted(res[c])
        if not seeds:
            continue
        L += ['', f'## H{k} 조건 {c} ({MAIN[c]}) 대 같은 seed e12 A + one 3.5', '',
              '| seed | 공기량 (e12 A) | 하루 비용 (e12 A) | 성립 |', '| --- | --- | --- | --- |']
        ok, airs, costs = [], [], []
        for s in seeds:
            r = res[c][s][MAIN[c]]
            airs.append(r['air_m3'])
            costs.append(r['cost'])
            ok.append(s in base and r['air_m3'] < base[s]['air_m3'] and r['cost'] < base[s]['cost'])
            b = base.get(s, {'air_m3': float('nan'), 'cost': float('nan')})
            L.append(f"| {s} | {r['air_m3']:.0f} ({b['air_m3']:.0f}) | {r['cost']:.2f} ({b['cost']:.2f}) | {'예' if ok[-1] else '아니오'} |")
        ma, mc = float(np.mean(airs)), float(np.mean(costs))
        ba, bc = float(np.mean([base[s]['air_m3'] for s in base])), float(np.mean([base[s]['cost'] for s in base]))
        sup = sum(ok) >= 3 and ma < ba and mc < bc
        L += ['', f"- H{k} 판정: 평균 공기량 {ma:.0f} 대 {ba:.0f}, 평균 비용 {mc:.2f} 대 {bc:.2f}, 성립 seed {sum(ok)}/{len(seeds)}: {'지지' if sup else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--cond', choices=CONDS, required=True)
    o.add_argument('--seed', type=int, required=True)
    for k in ('data', 'surrogate', 'runs', 'e12-runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'runs', 'e12', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
