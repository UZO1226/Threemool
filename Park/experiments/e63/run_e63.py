"""Park/e63 한 스텝 결정 계산 시간 (Park 작성). 규칙은 README.md 를 따른다.
현장 한 곳 (에피소드 1개) 을 가정하고, 검증 5일 x 구성원 0 에서 스텝마다 행동을 정하는 데 걸린 벽시계 시간을 잰다.

    python run_e63.py run --data .. --surrogate .. --runs <결과> --e60 <결과>/e60 --e54 <결과>/e54 --out <결과>/e63
    python run_e63.py report --out <결과>/e63 --md results.md
"""
import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import EnsemblePolicy, MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, load  # noqa: E402
from train_darrow import agent_act  # noqa: E402

REPEAT = 3       # 같은 측정을 3번 하고 스텝 시간을 모두 모은다
DAYS = 5         # 검증 40일에서 8일 간격 5일
NAMES = {'rep': '대표 (FCSRL 4 x 16 + 대표 차단, 묶음 0_1_2_3)', 'mpc_brake': '차단 위 예측 제어 (Park/e60 고름)',
         'mpc': '예측 제어만 (같은 설정, 차단 없음)', 'abac': '다시 고른 ABAC (Park/e54)'}


def methods(a, env, dyn):
    s60 = json.loads((Path(a.e60) / 'pick.json').read_text())
    s54 = json.loads((Path(a.e54) / 'pick.json').read_text())
    ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in COMBOS[0]])
    br = lambda: MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE)  # noqa: E731
    mpc = lambda: ShortMPC(env, dyn, margin=s60['margin'], lam=s60['lam'], objective='excess')  # noqa: E731
    return {'rep': (ens, br()), 'mpc_brake': (mpc(), br()), 'mpc': (mpc(), None), 'abac': (ABAC(env, *s54['abac']), None)}


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    days = splits['val'][::8][:DAYS]
    out = {'machine': platform.platform(), 'processor': platform.processor()}
    for name, (act, br) in methods(a, env, dyn).items():
        ts = []
        for _ in range(REPEAT):
            for d in days:
                obs = env.reset([(d, 0)])
                done = False
                while not done:
                    t0 = time.perf_counter()
                    u = np.asarray(act(obs), dtype=float).reshape(1)
                    if br is not None:
                        u, _ = br(obs, u, env)
                    ts.append(time.perf_counter() - t0)
                    obs, _, _, done, _ = env.step(u)
        ts = np.array(ts) * 1000.0
        out[name] = {'median_ms': float(np.median(ts)), 'mean_ms': float(ts.mean()), 'p95_ms': float(np.percentile(ts, 95)), 'n_steps': int(len(ts))}
        print(name, out[name], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'timing.json').write_text(json.dumps(out, indent=2))


def report(a):
    r = json.loads((Path(a.out) / 'timing.json').read_text())
    L = [f"에피소드 1개 (현장 한 곳), 검증 {DAYS}일 x 구성원 0, {REPEAT}번 반복, 스텝 15분. 기계: {r['machine']}.", '',
         '| 방법 | 스텝 시간 중앙값 ms | 평균 ms | 95% 분위 ms | 스텝 수 |', '| --- | --- | --- | --- | --- |']
    for k, n in NAMES.items():
        L.append(f"| {n} | {r[k]['median_ms']:.2f} | {r[k]['mean_ms']:.2f} | {r[k]['p95_ms']:.2f} | {r[k]['n_steps']} |")
    ratio = r['mpc_brake']['median_ms'] / r['rep']['median_ms']
    L += ['', f"- H1 (대표의 스텝 시간 중앙값이 차단 위 예측 제어의 1/10 이하인가): 차단 위 예측 제어 / 대표 = {ratio:.1f} 배: {'지지' if ratio >= 10 else '지지 안 됨'}",
          f"- 참고: 두 방법 모두 스텝 간격 15분 (900,000 ms) 보다 훨씬 짧다 (차단 위 예측 제어 95% 분위 {r['mpc_brake']['p95_ms']:.1f} ms)"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    for k in ('data', 'surrogate', 'runs', 'e60', 'e54', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
