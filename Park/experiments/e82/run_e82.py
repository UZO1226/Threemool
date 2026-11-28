"""Park/e82 검증 구간 날짜 블록으로 본 대표 대 차단 위 예측 제어 (Park 작성). 규칙은 README.md 를 따른다.
차단 위 예측 제어 두 설정 (기본: NH4 여유 1.2, 공기 가중치 1.5 / Park/e79: 여유 1.8, 가중치 2.0) 을 스텝 기록과 함께 검증 구간에서 돌리고,
대표의 스텝 기록 (Park/e44) 과 날짜마다 짝지어 하루 비용 차이의 날짜 블록 95% 구간을 낸다.
판정은 세 갈래다: 구간이 0 아래 (예측 제어가 낮음), 0 위 (대표가 낮음), 0 포함 (차이 없음).

    python run_e82.py run --part <i> --nparts 2 --data .. --surrogate .. --out <결과>/e82
    python run_e82.py report --out <결과>/e82 --e44 <결과>/e44 --md results.md
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
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, key  # noqa: E402
from run_e49 import BLOCK, NBOOT, SEED, block_ci, iid_ci, series  # noqa: E402

# 이름: (표 이름, NH4 여유, 공기 가중치)
MPCS = {'mpc12': ('차단 위 예측 제어 기본 (여유 1.2, 가중치 1.5, Park/e58)', 1.2, 1.5),
        'mpc18': ('차단 위 예측 제어 (여유 1.8, 가중치 2.0, Park/e79)', 1.8, 2.0)}


def load(p):
    return dict(np.load(p))


def run_mpc(a, split):
    """MPCS 의 설정을 split 날짜에서 스텝 기록과 함께 돈다 (Park/e83 도 쓴다)."""
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    days = splits['val'] if split == 'val' else splits['train'][-40:]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name in list(MPCS)[a.part::a.nparts]:
        _, m, lam = MPCS[name]
        r, L = rollout_log(env, days, ShortMPC(env, dyn, H=4, margin=m, lam=lam, objective='excess'),
                           shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        np.savez_compressed(Path(a.out) / f'log_{name}.npz', **L)
        (Path(a.out) / f'{name}.json').write_text(json.dumps(r, indent=2))
        print(name, r, flush=True)


def three_way(lo, hi, low_name, high_name):
    return f'{low_name} 낮음 (구간이 0 아래)' if hi < 0 else (f'{high_name} 낮음 (구간이 0 위)' if lo > 0 else '차이 없음 (구간이 0 포함)')


def compare(rep_logs, out_dir, header):
    """대표 스텝 기록 (묶음 목록) 과 MPCS 스텝 기록을 날짜마다 짝지어 (예측 제어 빼기 대표) 의 블록 구간을 낸다."""
    days, rc = series(rep_logs, 'cost')
    _, ra = series(rep_logs, 'air')
    rng = np.random.default_rng(SEED)
    L = [header, '',
         '| 비교 (예측 제어 빼기 대표) | 공기량 차이 | 하루 비용 차이 | 블록 95% 구간 | 날짜 독립 95% 구간 (참고) | 예측 제어가 나은 날 / 같은 날 / 나쁜 날 | 판정 |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    res = {}
    for n, (label, _, _) in MPCS.items():
        g = [load(Path(out_dir) / f'log_{n}.npz')]
        d_days, mc = series(g, 'cost')
        _, ma = series(g, 'air')
        assert d_days == days
        d = mc - rc
        lo, hi = block_ci(d, rng)
        ilo, ihi = iid_ci(d, rng)
        res[n] = (float(d.mean()), lo, hi, float((ma - ra).mean() / ra.mean() * 100))
        L.append(f"| {label} | {(ma - ra).mean():+,.0f} ({res[n][3]:+.1f}%) | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {ilo:+.3f} ~ {ihi:+.3f} | "
                 f"{(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()} | {three_way(lo, hi, '예측 제어가', '대표가')} |")
    return L, res, (days, rc, ra)


def report(a):
    rep = [load(Path(a.e44) / f'log_{key(c)}.npz') for c in COMBOS]
    head = (f'검증 40일. 날짜마다 구성원 5개 평균 (대표는 묶음 5개도 평균) 을 낸 뒤 날짜 단위로 짝지어 뺀다. '
            f'블록 구간: 이어진 {BLOCK}일 원형 이동 블록 다시 뽑기 {NBOOT:,}번 (seed {SEED}) 의 95% 백분위 (Park/e49, e64 와 같음). 대표 스텝 기록은 Park/e44.')
    L, res, _ = compare(rep, a.out, head)
    L += ['', '## 판정', '']
    for tag, n in (('H1', 'mpc12'), ('H2', 'mpc18')):
        m, lo, hi, _ = res[n]
        L.append(f"- {tag} ({MPCS[n][0]} 가 대표보다 낮은가, 블록 구간이 0 아래): 차이 {m:+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: "
                 f"{'지지' if hi < 0 else ('반대 (대표가 낮음)' if lo > 0 else '차이 없음')}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=2)
    for k in ('data', 'surrogate', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e44', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_mpc(a, 'val') if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
