"""Park/e84 학습하지 않는 기준 (차단 + ABAC) 으로 학습한 날의 이점 가르기 (Park 작성). 규칙은 README.md 를 따른다.
대표 차단 위 ABAC (Kp 4,000, Ki 1,600, Park/e54, e70) 를 설정값 여러 개로 학습 구간 마지막 40일과 검증 구간에서 스텝 기록과 함께 돌린다.
대표와 차단 위 예측 제어 (기본) 를 같은 공기량의 차단 + ABAC (설정값 사이 날짜별 선형 보간) 와 날짜마다 짝지어 블록 95% 구간을 낸다.

    python run_e84.py run --part <i> --nparts 7 --data .. --surrogate .. --out <결과>/e84
    python run_e84.py report --out <결과>/e84 --e44 <결과>/e44 --e82 <결과>/e82 --e83 <결과>/e83 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
for d in ('e38', 'e49', 'e82'):
    sys.path.insert(0, str(HERE.parent / d))
from audit_brake import rollout_log  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from run_e38 import BRAKE, COMBOS, key  # noqa: E402
from run_e49 import BLOCK, NBOOT, SEED, block_ci, series  # noqa: E402
from run_e82 import load  # noqa: E402

KP, KI = 4000.0, 1600.0
JOBS = [('train', sp) for sp in (2.6, 2.9, 3.2, 3.8, 5.0)] + [('val', sp) for sp in (2.6, 3.2)]


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for split, sp in JOBS[a.part::a.nparts]:
        days = splits['val'] if split == 'val' else splits['train'][-40:]
        r, L = rollout_log(env, days, ABAC(env, sp, KP, KI), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        np.savez_compressed(Path(a.out) / f'log_{split}_{sp}.npz', **L)
        (Path(a.out) / f'{split}_{sp}.json').write_text(json.dumps(r, indent=2))
        print(split, sp, r, flush=True)


def matched(out, split, target, days):
    """같은 split 의 차단 + ABAC 점들을 전체 공기량 순으로 놓고, target 공기량을 감싸는 두 점의 날짜별 비용을 선형 보간한다.
    범위 밖이면 None."""
    pts = []
    for sp in sorted({s for sl, s in JOBS if sl == split}):
        g = [load(Path(out) / f'log_{split}_{sp}.npz')]
        d, c = series(g, 'cost')
        _, ar = series(g, 'air')
        assert d == days
        pts.append((ar.mean(), c, sp))
    pts.sort(key=lambda p: p[0])
    for (a0, c0, s0), (a1, c1, s1) in zip(pts, pts[1:]):
        if a0 <= target <= a1 and a1 > a0:
            w = (target - a0) / (a1 - a0)
            return (1 - w) * c0 + w * c1, f'설정값 {s1:g} 과 {s0:g} 사이 (w {w:.2f})', [(p[0], p[2]) for p in pts]
    return None, '범위 밖', [(p[0], p[2]) for p in pts]


def verdict(lo, hi, low_name):
    return f'{low_name} 낮음 (구간이 0 아래)' if hi < 0 else ('차단 + ABAC 가 낮음 (구간이 0 위)' if lo > 0 else '차이 없음 (구간이 0 포함)')


def report(a):
    rng = np.random.default_rng(SEED)
    rep_val = [load(Path(a.e44) / f'log_{key(c)}.npz') for c in COMBOS]
    rep_tr = [load(Path(a.e83) / f'log_rep_{key(c)}.npz') for c in COMBOS]
    cases = [('train', '대표', rep_tr), ('train', '차단 위 예측 제어 기본', [load(Path(a.e83) / 'log_mpc12.npz')]),
             ('val', '대표', rep_val), ('val', '차단 위 예측 제어 기본', [load(Path(a.e82) / 'log_mpc12.npz')])]
    L = [f'차단 + ABAC (Kp {KP:,.0f}, Ki {KI:,.0f}, 대표 차단) 는 학습하지 않으므로 학습한 날의 이점이 없다. 각 방법의 전체 공기량에서 차단 + ABAC 의 날짜별 비용을 설정값 사이 선형 보간하고, '
         f'(방법 빼기 차단 + ABAC) 를 날짜 단위로 짝짓는다. 블록 구간: 이어진 {BLOCK}일 원형 이동 블록 다시 뽑기 {NBOOT:,}번 (seed {SEED}) 의 95% 백분위.', '']
    for split in ('train', 'val'):
        pts = []
        for sp in sorted({s for sl, s in JOBS if sl == split}):
            r = json.loads((Path(a.out) / f'{split}_{sp}.json').read_text())
            pts.append(f"설정값 {sp:g}: {r['air_m3']:,.0f} / {r['cost']:.3f}")
        L.append(f"- 차단 + ABAC ({'학습 구간 마지막 40일' if split == 'train' else '검증 40일'}): " + ', '.join(pts))
    L += ['', '| 날 | 방법 | 공기량 | 하루 비용 | 같은 공기량의 차단 + ABAC 비용 | 보간 | 차이 (방법 빼기 차단 + ABAC) | 블록 95% 구간 | 판정 |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    R = {}
    for split, name, logs in cases:
        days, c = series(logs, 'cost')
        _, ar = series(logs, 'air')
        m, how, _ = matched(a.out, split, ar.mean(), days)
        if m is None:
            R[(split, name)] = None
            L.append(f"| {'학습 구간' if split == 'train' else '검증'} | {name} | {ar.mean():,.0f} | {c.mean():.3f} | 범위 밖 | | | | 판정 안 함 |")
            continue
        d = c - m
        lo, hi = block_ci(d, rng)
        R[(split, name)] = (float(d.mean()), lo, hi)
        L.append(f"| {'학습 구간' if split == 'train' else '검증'} | {name} | {ar.mean():,.0f} | {c.mean():.3f} | {m.mean():.3f} | {how} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {verdict(lo, hi, name + ('가' if name == '대표' else '이'))} |")

    def v(k, ref=False):
        r = R[k]
        if r is None:
            return '판정 안 함 (범위 밖)'
        m, lo, hi = r
        return f"차이 {m:+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: " + (('0 아래' if ref else '지지') if hi < 0 else ('반대 (차단 + ABAC 가 낮음)' if lo > 0 else '차이 없음'))
    L += ['', '## 판정', '',
          f"- H1 (학습 구간 마지막 40일에서 차단 위 예측 제어 기본이 같은 공기량의 차단 + ABAC 보다 낮은가, 구간이 0 아래): {v(('train', '차단 위 예측 제어 기본'))}",
          f"- H2 (학습 구간 마지막 40일에서 대표가 같은 공기량의 차단 + ABAC 보다 낮은가, 구간이 0 아래): {v(('train', '대표'))}",
          f"- 참고 (판정 규칙의 짝, 검증 40일): 예측 제어 기본 {v(('val', '차단 위 예측 제어 기본'), True)} / 대표 {v(('val', '대표'), True)}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=7)
    for k in ('data', 'surrogate', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e44', 'e82', 'e83', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
