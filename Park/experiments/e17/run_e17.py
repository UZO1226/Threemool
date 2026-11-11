"""Park/e17 회복 가능성 검사를 더한 차단 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e17.py one --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --e12-runs <결과>/dw_minbrake_runs --out <결과>/e17
    python run_e17.py report --out <결과>/e17 --e12 <결과>/e12 --md results.md
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
from min_brake import E15_MARGINS, MinAirBrake  # noqa: E402
from run_e07 import ms  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BASE = dict(trim=True, margins=P1, do_limit=1.8)
SETTINGS = {  # 이름: MinAirBrake 인자 (target 3.5, 1스텝, 다른 구성원)
    'p1_do1.8': BASE,
    'rec3_3.5': dict(BASE, recover=(3, 3.5)),
    'rec9_3.5': dict(BASE, recover=(9, 3.5)),
    'rec3_4.0': dict(BASE, recover=(3, 4.0)),
}
JUDGE = ['rec3_3.5', 'rec9_3.5', 'rec3_4.0']   # H1, H2, H3 (README 에 미리 적음)


def run_one(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    agent = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    agent.load(Path(a.e12_runs) / f'A_s{a.seed}' / 'model_final')
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    out = {'seed': a.seed}
    for name, kw in SETTINGS.items():
        brake = MinAirBrake(dyn, 3.5, 1, 'others', **kw)
        out[name] = evaluate(env, splits['val'], act, shield=brake)
        out[name]['lowered_pct'] = 100 * brake.lowered_steps / max(brake.steps, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'seed_{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.seed, flush=True)


def report(a):
    res = {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(a.out).glob('seed_*.json'))}
    e12 = {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(a.e12).glob('A_seed_*.json'))}
    base = {s: e12[s]['min1']['3.5'] for s in e12}
    seeds = sorted(res)
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % | 내린 % |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} | "
                f"{ms([p.get('lowered_pct', 0) for p in ps], 1)} |")
    L = ['e12 A 정책 seed 5개, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', ''] + head
    L.append(row('기준: e12 A + 한 방향 3.5', [base[s] for s in seeds]))
    L += [row(n, [res[s][n] for s in seeds]) for n in SETTINGS]
    L += ['', '## seed 평균 비용 <= 12 (PROTOCOL v1)', '']
    L += [f"- seed {s}: " + ', '.join(f"{n} {res[s][n]['cost']:.2f}" for n in JUDGE) for s in seeds]
    for k, n in enumerate(JUDGE, 1):
        L += ['', f'## H{k} {n} 대 같은 seed e12 A + 한 방향 3.5', '', '| seed | 공기량 (기준) | 하루 비용 (기준) | 성립 |', '| --- | --- | --- | --- |']
        ok = []
        for s in seeds:
            r, b = res[s][n], base[s]
            ok.append(r['air_m3'] < b['air_m3'] and r['cost'] < b['cost'])
            L.append(f"| {s} | {r['air_m3']:.0f} ({b['air_m3']:.0f}) | {r['cost']:.2f} ({b['cost']:.2f}) | {'예' if ok[-1] else '아니오'} |")
        ma, mc = np.mean([res[s][n]['air_m3'] for s in seeds]), np.mean([res[s][n]['cost'] for s in seeds])
        ba, bc = np.mean([base[s]['air_m3'] for s in seeds]), np.mean([base[s]['cost'] for s in seeds])
        sup = sum(ok) >= 3 and ma < ba and mc < bc
        L += ['', f"- H{k} 판정: 평균 공기량 {ma:.0f} 대 {ba:.0f}, 평균 비용 {mc:.2f} 대 {bc:.2f}, 성립 seed {sum(ok)}/{len(seeds)}: {'지지' if sup else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--seed', type=int, required=True)
    for k in ('data', 'surrogate', 'e12-runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e12', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()
