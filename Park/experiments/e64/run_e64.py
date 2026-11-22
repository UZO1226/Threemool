"""Park/e64 새 기준에 대한 날짜 블록 불확실성 (Park 작성). 규칙은 README.md 를 따른다.
다시 고른 ABAC (Park/e54) 와 차단 위 예측 제어 (Park/e60) 를 스텝 기록과 함께 검증 구간에서 돌리고, 대표의 스텝 기록 (Park/e44) 과
Park/e49 와 같은 이어진 5일 원형 이동 블록 다시 뽑기로 날짜 평균 차이의 구간을 낸다.

    python run_e64.py run --part <i> --nparts 2 --data .. --surrogate .. --e54 <결과>/e54 --e60 <결과>/e60 --out <결과>/e64
    python run_e64.py report --out <결과>/e64 --e44 <결과>/e44 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e49'))
from audit_brake import rollout_log  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, key  # noqa: E402
from run_e49 import BLOCK, NBOOT, SEED, block_ci, iid_ci, series  # noqa: E402

JOBS = ('abac54', 'mpc_brake60')
NAMES = {'abac54': '다시 고른 ABAC (Park/e54)', 'mpc_brake60': '차단 위 예측 제어 (Park/e60)'}


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name in JOBS[a.part::a.nparts]:
        if name == 'abac54':
            r, L = rollout_log(env, splits['val'], ABAC(env, *json.loads((Path(a.e54) / 'pick.json').read_text())['abac']))
        else:
            s = json.loads((Path(a.e60) / 'pick.json').read_text())
            r, L = rollout_log(env, splits['val'], ShortMPC(env, dyn, margin=s['margin'], lam=s['lam'], objective='excess'),
                               shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        np.savez_compressed(Path(a.out) / f'log_{name}.npz', **L)
        (Path(a.out) / f'e64_{name}.json').write_text(json.dumps(r, indent=2))
        print(name, r, flush=True)


def report(a):
    load = lambda p: dict(np.load(p))  # noqa: E731
    R = [load(Path(a.e44) / f'log_{key(c)}.npz') for c in COMBOS]
    days, rc = series(R, 'cost')
    _, ra = series(R, 'air')
    rng = np.random.default_rng(SEED)
    L = [f'검증 {len(days)}일. 날짜마다 구성원 5개 평균 (대표는 묶음 5개도 평균) 을 낸 뒤 날짜 단위로 짝지어 뺀다. 블록 구간: 이어진 {BLOCK}일 원형 이동 블록 다시 뽑기 {NBOOT:,}번 (seed {SEED}) 의 95% 백분위 (Park/e49 와 같음).', '',
         '| 비교 (대표 빼기) | 하루 비용 차이 | 블록 95% 구간 | 날짜 독립 95% 구간 (참고) | 대표가 나은 날 / 같은 날 / 나쁜 날 | 공기량 차이 |', '| --- | --- | --- | --- | --- | --- |']
    D = {}
    for n in JOBS:
        g = [load(Path(a.out) / f'log_{n}.npz')]
        _, bc = series(g, 'cost')
        _, ba = series(g, 'air')
        d = rc - bc
        lo, hi = block_ci(d, rng)
        ilo, ihi = iid_ci(d, rng)
        D[n] = (d, lo, hi)
        L.append(f"| {NAMES[n]} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {ilo:+.3f} ~ {ihi:+.3f} | {(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()} | {(ra - ba).mean():+,.0f} |")
    d, lo, hi = D['abac54']
    L += ['', '## 판정', '', f"- H1 (대표 대 다시 고른 ABAC): 차이 {d.mean():+.3f}, 블록 95% 구간 {lo:+.3f} ~ {hi:+.3f}: {'지지' if hi < 0 else '지지 안 됨'}"]
    d, lo, hi = D['mpc_brake60']
    v = '대표가 낮음 (구간이 0 아래)' if hi < 0 else ('차단 위 예측 제어가 낮음 (구간이 0 위)' if lo > 0 else '차이 없음 (구간이 0 포함)')
    L.append(f"- H2 (대표 대 차단 위 예측 제어, 방향을 정하지 않은 구분): 차이 {d.mean():+.3f}, 블록 95% 구간 {lo:+.3f} ~ {hi:+.3f}: {v}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=2)
    for k in ('data', 'surrogate', 'e54', 'e60', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e44', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
