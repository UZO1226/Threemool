"""Park/e20 정책 앙상블 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e20.py run --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --e12-runs <결과>/dw_minbrake_runs --out <결과>/e20
    python run_e20.py report --out <결과>/e20 --e19 <결과>/e19 --md results.md
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
COMBINE = {  # 이름: seed 5개 정책 행동 (5, N) 을 하나로 합치는 방법
    'median': lambda A: np.median(A, axis=0),
    'mean': lambda A: A.mean(axis=0),
    'q75': lambda A: np.quantile(A, 0.75, axis=0),
}
JUDGE = ['median', 'mean', 'q75']   # H1, H2, H3 (README 에 미리 적음)


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    agents = []
    for s in range(5):
        ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
        ag.load(Path(a.e12_runs) / f'A_s{s}' / 'model_final')
        agents.append(ag)
    out = {}
    for name, comb in COMBINE.items():
        act = lambda obs, comb=comb: comb(np.stack([np.asarray(agent_act(g, obs), float).reshape(len(obs)) for g in agents]))  # noqa: E731
        brake = MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE)
        out[name] = evaluate(env, splits['val'], act, shield=brake)
        print(name, out[name], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'ensemble.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = json.loads((Path(a.out) / 'ensemble.json').read_text())
    e19 = {r['seed']: r['conf_l3.5_m0.1_rec'] for r in (json.loads(p.read_text()) for p in Path(a.e19).glob('seed_*.json'))}
    seeds = sorted(e19)
    ba, bc = np.mean([e19[s]['air_m3'] for s in seeds]), np.mean([e19[s]['cost'] for s in seeds])
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 앙상블은 결정적 제어기 하나라 값이 하나다. 기준은 e19 H1 seed 5개.', '',
         '| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |',
         f"| 기준: e19 H1 seed 평균 | {ba:.0f} | {bc:.2f} | {np.mean([e19[s]['nh4_viol_pct'] for s in seeds]):.2f} | "
         f"{np.mean([e19[s]['do_viol_pct'] for s in seeds]):.2f} | {np.mean([e19[s]['days_within_limit_pct'] for s in seeds]):.1f} | "
         f"{np.mean([e19[s]['intervention_pct'] for s in seeds]):.1f} |"]
    for n in COMBINE:
        r = res[n]
        L.append(f"| 앙상블 {n} | {r['air_m3']:.0f} | {r['cost']:.2f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | "
                 f"{r['days_within_limit_pct']:.1f} | {r['intervention_pct']:.1f} |")
    for k, n in enumerate(JUDGE, 1):
        r = res[n]
        L += ['', f'## H{k} 앙상블 {n} 대 e19 H1 seed 별', '', '| seed | e19 H1 공기량 | e19 H1 하루 비용 | 앙상블이 둘 다 낮음 |', '| --- | --- | --- | --- |']
        ok = []
        for s in seeds:
            b = e19[s]
            ok.append(r['air_m3'] < b['air_m3'] and r['cost'] < b['cost'])
            L.append(f"| {s} | {b['air_m3']:.0f} | {b['cost']:.2f} | {'예' if ok[-1] else '아니오'} |")
        sup = sum(ok) >= 3 and r['air_m3'] < ba and r['cost'] < bc
        L += ['', f"- H{k} 판정: 공기량 {r['air_m3']:.0f} 대 평균 {ba:.0f}, 비용 {r['cost']:.2f} 대 평균 {bc:.2f}, 이긴 seed {sum(ok)}/{len(seeds)}: {'지지' if sup else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    for k in ('data', 'surrogate', 'e12-runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e19', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
