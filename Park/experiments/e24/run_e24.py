"""Park/e24 정책 학습률 6e-4 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e24.py run --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e24
    python run_e24.py report --out <결과>/e24 --e19 <결과>/e19 --e23 <결과>/e23 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1)   # e19 H1 차단
SEEDS = list(range(10))
ENS10 = (81376, 4.80)   # e23 README: 4 x 16 앙상블 조합 10개 평균 (H2 기준)


def load(path):
    ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(path)
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for s in SEEDS:
        ag = load(Path(a.runs) / 'dw_lr6_runs' / f'A_s{s}' / 'model_final')
        out[s] = evaluate(env, splits['val'], lambda obs: agent_act(ag, obs), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print('lr6e-4', s, out[s], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'e24.json').write_text(json.dumps(out, indent=2))


def report(a):
    new = {int(s): r for s, r in json.loads((Path(a.out) / 'e24.json').read_text()).items()}
    base = {r['seed']: r['conf_l3.5_m0.1_rec'] for r in (json.loads(p.read_text()) for p in Path(a.e19).glob('seed_*.json'))}
    base.update({int(s): r for s, r in json.loads((Path(a.e23) / 'e23.json').read_text())['single'].items()})
    assert sorted(base) == SEEDS
    mean = lambda d, k: float(np.mean([d[s][k] for s in SEEDS]))  # noqa: E731
    sd = lambda d, k: float(np.std([d[s][k] for s in SEEDS]))  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드, seed 10개. 둘 다 e19 H1 차단.', '',
         '| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for label, d in (('기준 e12 A (학습률 3e-4)', base), ('학습률 6e-4', new)):
        L.append(f"| {label} (평균 ± 표준편차) | {mean(d, 'air_m3'):.0f} ± {sd(d, 'air_m3'):.0f} | {mean(d, 'cost'):.2f} ± {sd(d, 'cost'):.2f} | "
                 f"{mean(d, 'nh4_viol_pct'):.2f} | {mean(d, 'do_viol_pct'):.2f} | {mean(d, 'days_within_limit_pct'):.1f} | {mean(d, 'intervention_pct'):.1f} |")
    L += ['', '## seed 별', '', '| seed | 기준 공기량 | 기준 비용 | 6e-4 공기량 | 6e-4 비용 | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    ok = []
    for s in SEEDS:
        b, n = base[s], new[s]
        ok.append(n['air_m3'] < b['air_m3'] and n['cost'] < b['cost'])
        L.append(f"| {s} | {b['air_m3']:.0f} | {b['cost']:.2f} | {n['air_m3']:.0f} | {n['cost']:.2f} | {'예' if ok[-1] else '아니오'} |")
    ma, mc = mean(new, 'air_m3'), mean(new, 'cost')
    sup1 = sum(ok) >= 6 and ma < mean(base, 'air_m3') and mc < mean(base, 'cost')
    sup2 = ma < ENS10[0] and mc < ENS10[1]
    L += ['', f"- H1 판정: 둘 다 낮은 seed {sum(ok)}/10, 공기량 {ma:.0f} 대 {mean(base, 'air_m3'):.0f}, 비용 {mc:.2f} 대 {mean(base, 'cost'):.2f}: {'지지' if sup1 else '지지 안 됨'}",
          f"- H2 판정: 공기량 {ma:.0f} 대 {ENS10[0]}, 비용 {mc:.2f} 대 {ENS10[1]:.2f}: {'지지' if sup2 else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e19', 'e23', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
