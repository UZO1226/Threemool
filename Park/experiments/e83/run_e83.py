"""Park/e83 학습 구간 마지막 40일 날짜 블록으로 본 대표 대 차단 위 예측 제어 (Park 작성). 규칙은 README.md 를 따른다.
Park/e82 와 같은 두 설정과 대표 (묶음 5개) 를 학습 구간 마지막 40일에서 스텝 기록과 함께 돌리고,
날짜마다 짝지어 (예측 제어 빼기 대표) 하루 비용 차이의 날짜 블록 95% 구간을 낸다 (세 갈래 판정).

    python run_e83.py run --part <i> --nparts 7 --data .. --surrogate .. --runs <runs> --out <결과>/e83
    python run_e83.py report --out <결과>/e83 --e43 <결과>/e43 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
for d in ('e38', 'e43', 'e49', 'e82'):
    sys.path.insert(0, str(HERE.parent / d))
from audit_brake import rollout_log  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, key  # noqa: E402
from run_e43 import rep  # noqa: E402
from run_e49 import BLOCK, NBOOT, SEED  # noqa: E402
from run_e82 import MPCS, compare, load  # noqa: E402

JOBS = list(MPCS) + [f'rep_{key(c)}' for c in COMBOS]


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    days = splits['train'][-40:]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name in JOBS[a.part::a.nparts]:
        if name in MPCS:
            _, m, lam = MPCS[name]
            act = ShortMPC(env, dyn, H=4, margin=m, lam=lam, objective='excess')
        else:
            act = rep(a.runs, next(c for c in COMBOS if f'rep_{key(c)}' == name))
        r, L = rollout_log(env, days, act, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        np.savez_compressed(Path(a.out) / f'log_{name}.npz', **L)
        (Path(a.out) / f'{name}.json').write_text(json.dumps(r, indent=2))
        print(name, r, flush=True)


def report(a):
    reps = [load(Path(a.out) / f'log_rep_{key(c)}.npz') for c in COMBOS]
    # 대표 스텝 기록이 Park/e43 의 학습 구간 값과 같은지 확인 (같은 정책, 같은 차단, 같은 날, 결정적)
    e43 = {}
    for p in sorted(Path(a.e43).glob('rep_train_part*.json')):
        e43.update(json.loads(p.read_text()))
    chk = []
    for c in COMBOS:
        r = json.loads((Path(a.out) / f'rep_{key(c)}.json').read_text())
        chk.append(abs(r['cost'] - e43[key(c)]['cost']) < 1e-9 and abs(r['air_m3'] - e43[key(c)]['air_m3']) < 1e-3)
    head = (f'학습 구간 마지막 40일. 날짜마다 구성원 5개 평균 (대표는 묶음 5개도 평균) 을 낸 뒤 날짜 단위로 짝지어 뺀다. '
            f'블록 구간: 이어진 {BLOCK}일 원형 이동 블록 다시 뽑기 {NBOOT:,}번 (seed {SEED}) 의 95% 백분위. '
            f'대표 스텝 기록이 Park/e43 학습 구간 값과 같은 묶음: {sum(chk)}/5.')
    L, res, _ = compare(reps, a.out, head)
    L += ['', '## 판정', '']
    for tag, n in (('H1', 'mpc12'), ('H2', 'mpc18')):
        m, lo, hi, _ = res[n]
        L.append(f"- {tag} (이 날에서 대표가 {MPCS[n][0]} 보다 낮은가, 블록 구간이 0 위): 차이 {m:+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: "
                 f"{'지지' if lo > 0 else ('반대 (예측 제어가 낮음)' if hi < 0 else '차이 없음')}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=7)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e43', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
