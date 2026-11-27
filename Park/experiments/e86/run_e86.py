"""Park/e86 차단의 몫을 날짜 블록으로 (검증 40일, 학습 구간 188일) (Park 작성). 규칙은 README.md 를 따른다.
같은 ABAC (설정값 3.2, Kp 4,000, Ki 1,600, Park/e54) 에 대표 차단을 씌운 것과 씌우지 않은 것을 날짜마다 짝지어 하루 비용 차이의 날짜 블록 95% 구간을 낸다.
검증 40일은 기존 스텝 기록 (차단 없음: Park/e64, 차단: Park/e84) 을 쓰고, 학습 구간 188일은 새로 돈다 (날짜를 4 조각으로 나눠 동시에 돈다).
이 파일의 도움 함수 (조각 실행, 기록 합치기, 공기량 맞추기, 세 갈래 판정) 는 Park/e87 ~ e94 도 쓴다.

    python run_e86.py run --part <i> --nparts 8 --data .. --surrogate .. --out <결과>/e86
    python run_e86.py report --out <결과>/e86 --e64 <결과>/e64 --e84 <결과>/e84 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
for d in ('e38', 'e49'):
    sys.path.insert(0, str(HERE.parent / d))
from audit_brake import rollout_log  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from run_e38 import BRAKE  # noqa: E402
from run_e49 import BLOCK, NBOOT, SEED, block_ci, series  # noqa: E402

KP, KI, SP = 4000.0, 1600.0, 3.2
NCHUNK = 4


# ---- 도움 함수 (Park/e87 ~ e94 도 씀) ----
def setup(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    return splits, BatchEnv(day_data, dyn), dyn


def days_of(splits, which):
    return {'val': splits['val'], 'train': splits['train'], 'trainlast': splits['train'][-40:]}[which]


def brake(dyn, **kw):
    return MinAirBrake(dyn, 3.5, 1, 'others', **dict(BRAKE, **kw))


def run_chunks(a, jobs, make):
    """jobs: [(이름, 날 종류)]. 날을 NCHUNK 조각으로 나눈 (이름, 조각) 을 part 로 나눠 돈다. make(env, dyn, 이름) -> (act, shield)."""
    splits, env, dyn = setup(a)
    tasks = [(n, w, c) for n, w in jobs for c in range(NCHUNK if w == 'train' else 1)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for n, w, c in tasks[a.part::a.nparts]:
        days = days_of(splits, w)
        if w == 'train':
            days = days[c::NCHUNK]
        act, sh = make(env, dyn, n)
        r, L = rollout_log(env, days, act, shield=sh)
        np.savez_compressed(Path(a.out) / f'log_{n}_c{c}.npz', **L)
        (Path(a.out) / f'{n}_c{c}.json').write_text(json.dumps(r, indent=2))
        print(n, c, r, flush=True)
    return len(tasks)


def load_log(out, name):
    """조각 기록을 합친다 (조각이 하나면 그대로). log_<name>.npz 하나만 있어도 읽는다."""
    ps = sorted(Path(out).glob(f'log_{name}_c*.npz')) or [Path(out) / f'log_{name}.npz']
    gs = [dict(np.load(p)) for p in ps]
    return {k: np.concatenate([g[k] for g in gs]) for k in gs[0]}


def daily(logs):
    """기록 목록 (묶음들) 의 날짜별 비용과 공기량 (구성원과 묶음 평균)."""
    d, c = series(logs, 'cost')
    _, a = series(logs, 'air')
    return d, c, a


def matched(points, target):
    """points: [(전체 공기량, 날짜별 비용, 이름)]. target 공기량을 감싸는 두 점의 날짜별 비용을 선형 보간. 범위 밖이면 (None, '범위 밖')."""
    pts = sorted(points, key=lambda p: p[0])
    for (a0, c0, s0), (a1, c1, s1) in zip(pts, pts[1:]):
        if a0 <= target <= a1 and a1 > a0:
            w = (target - a0) / (a1 - a0)
            return (1 - w) * c0 + w * c1, f'{s0} 와 {s1} 사이 (비중 {w:.2f})'
    return None, '범위 밖'


def ci(d):
    """날짜 블록 95% 구간 (seed 고정, 비교마다 같은 난수열)."""
    return block_ci(np.asarray(d), np.random.default_rng(SEED))


def three(lo, hi, below='지지', above='반대'):
    return below if hi < 0 else (above if lo > 0 else '차이 없음')


BLOCK_NOTE = f'날짜 블록 구간: 이어진 {BLOCK}일 원형 이동 블록 다시 뽑기 {NBOOT:,}번 (seed {SEED}) 의 95% 백분위 (Park/e49, e82 와 같음).'


# ---- e86 ----
def make(env, dyn, n):
    return ABAC(env, SP, KP, KI), (brake(dyn) if n == 'abac_b' else None)


def report(a):
    rows = {'val': (load_log(a.e64, 'abac54'), load_log(a.e84, 'val_3.2')),
            'train': (load_log(a.out, 'abac_nb'), load_log(a.out, 'abac_b'))}
    L = [f'ABAC (설정값 {SP}, Kp {KP:,.0f}, Ki {KI:,.0f}) 에 대표 차단을 씌운 것 빼기 씌우지 않은 것. 날짜마다 구성원 5개 평균. {BLOCK_NOTE}', '',
         '| 날 | 날 수 | 차단 없음 공기량 / 비용 | 차단 공기량 / 비용 | 공기량 차이 (블록 95% 구간) | 하루 비용 차이 | 블록 95% 구간 | 차단이 나은 날 / 같은 날 / 나쁜 날 | 판정 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    R = {}
    for w, (nb, b) in rows.items():
        d0, c0, a0 = daily([nb])
        d1, c1, a1 = daily([b])
        assert d0 == d1
        dc, da = c1 - c0, a1 - a0
        lo, hi = ci(dc)
        alo, ahi = ci(da)
        R[w] = (dc.mean(), lo, hi)
        L.append(f"| {'검증' if w == 'val' else '학습 구간 전체'} | {len(d0)} | {a0.mean():,.0f} / {c0.mean():.3f} | {a1.mean():,.0f} / {c1.mean():.3f} | "
                 f"{da.mean():+,.0f} ({alo:+,.0f} ~ {ahi:+,.0f}) | {dc.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {(dc < 0).sum()} / {(dc == 0).sum()} / {(dc > 0).sum()} | {three(lo, hi)} |")
    L += ['', '## 판정', '']
    for tag, w, what in (('H1', 'val', '검증 40일'), ('H2', 'train', '학습 구간 188일')):
        m, lo, hi = R[w]
        L.append(f'- {tag} ({what} 에서 차단이 같은 ABAC 의 하루 비용을 낮추는가, 블록 구간이 0 아래): 차이 {m:+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: {three(lo, hi)}')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=8)
    for k in ('data', 'surrogate', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e64', 'e84', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        run_chunks(a, [('abac_nb', 'train'), ('abac_b', 'train')], make)
    else:
        report(a)


if __name__ == '__main__':
    main()
