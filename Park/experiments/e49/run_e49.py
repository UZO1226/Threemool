"""Park/e49 날짜 블록 불확실성 (Park 작성). 규칙은 README.md 를 따른다. 새 평가는 하지 않고 Park/e44, e45 의 스텝 기록만 쓴다.

    python run_e49.py report --e44 <결과>/e44 --e45 <결과>/e45 --md results.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from darrow_env import STEP_H  # noqa: E402
from run_e38 import COMBOS, ci, key  # noqa: E402

BLOCK = 5        # 이어진 5일을 한 블록으로 다시 뽑는다 (40일 = 8 블록)
NBOOT = 10000    # 다시 뽑기 횟수
SEED = 0         # 다시 뽑기 난수 seed (고정)
DROP = 4         # H3: 대표 이득이 가장 큰 4일 (10%) 을 뺀다


def per_day(g, k):
    """스텝 기록 하나의 날짜별 값 (구성원 5개 평균). k 는 'cost' 또는 'air'."""
    v = (g['v_nh4'] + g['v_do']).sum(axis=1) if k == 'cost' else g['air'].sum(axis=1) * STEP_H
    days = sorted(set(g['day']))
    return days, np.array([v[g['day'] == d].mean() for d in days])


def series(gs, k):
    out = [per_day(g, k) for g in gs]
    days = out[0][0]
    assert all(o[0] == days for o in out)
    return days, np.mean([o[1] for o in out], axis=0)


def block_ci(diff, rng):
    """원형 이동 블록 다시 뽑기로 날짜 평균 차이의 95% 백분위 구간."""
    n = len(diff)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(NBOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(NBOOT, -1)[:, :n] % n
    m = diff[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def iid_ci(diff, rng):
    """날짜를 하나씩 독립으로 다시 뽑은 95% 백분위 구간 (참고, 블록 효과 비교용)."""
    n = len(diff)
    m = diff[rng.integers(0, n, size=(NBOOT, n))].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def report(a):
    load = lambda p: dict(np.load(p))  # noqa: E731
    R = [load(Path(a.e44) / f'log_{key(c)}.npz') for c in COMBOS]
    base = {'abac': [load(Path(a.e45) / 'log_abac.npz')], 'brake': [load(Path(a.e45) / 'log_brake.npz')],
            'ppo': [load(Path(a.e45) / f'log_ppo_{s}.npz') for s in range(5)],
            'fcsrl': [load(Path(a.e45) / f'log_fcsrl_{s}.npz') for s in range(5)]}
    names = {'abac': 'ABAC (Park/e43 고름)', 'brake': '차단만 (Park/e43 고름)', 'ppo': '기본 PPO-Lag, 차단 없음 (seed 5개)', 'fcsrl': '기본 FCSRL, 차단 없음 (seed 5개)'}
    days, rc = series(R, 'cost')
    _, ra = series(R, 'air')
    rng = np.random.default_rng(SEED)
    L = [f'검증 {len(days)}일 ({days[0]} ~ {days[-1]}). 날짜마다 구성원 5개 평균 (대표는 묶음 5개도 평균) 을 낸 뒤 날짜 단위로 짝지어 뺀다. '
         f'블록 구간: 이어진 {BLOCK}일 원형 이동 블록 다시 뽑기 {NBOOT:,}번 (seed {SEED}) 의 95% 백분위.', '',
         '| 비교 (대표 빼기) | 하루 비용 차이 | 블록 95% 구간 | 날짜 독립 95% 구간 (참고) | 대표가 나은 날 / 같은 날 / 나쁜 날 | 공기량 차이 | 공기량 블록 95% 구간 |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    D = {}
    for b, gs in base.items():
        _, bc = series(gs, 'cost')
        _, ba = series(gs, 'air')
        d, da = rc - bc, ra - ba
        lo, hi = block_ci(d, rng)
        ilo, ihi = iid_ci(d, rng)
        alo, ahi = block_ci(da, rng)
        D[b] = (d, lo, hi)
        L.append(f"| {names[b]} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {ilo:+.3f} ~ {ihi:+.3f} | {(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()} | {da.mean():+,.0f} | {alo:+,.0f} ~ {ahi:+,.0f} |")
    L += ['', '## 판정', '']
    for hn, b in (('H1', 'abac'), ('H2', 'brake')):
        d, lo, hi = D[b]
        L.append(f"- {hn} (대표 대 {names[b]}): 날짜 평균 차이 {d.mean():+.3f}, 블록 95% 구간 {lo:+.3f} ~ {hi:+.3f}: {'지지' if hi < 0 else '지지 안 됨'}")
    d = D['abac'][0]
    keep = np.sort(np.argsort(d, kind='stable')[DROP:])
    lo, hi = block_ci(d[keep], rng)
    drop = sorted(days[i] for i in np.argsort(d, kind='stable')[:DROP])
    L.append(f"- H3 (이득이 가장 큰 {DROP}일 {', '.join(drop)} 을 뺀 대표 대 ABAC, {len(keep)}일): 평균 차이 {d[keep].mean():+.3f}, 블록 95% 구간 {lo:+.3f} ~ {hi:+.3f}: {'지지' if d[keep].mean() < 0 and hi < 0 else '지지 안 됨'}")
    share = -d[np.argsort(d, kind='stable')[:DROP]].sum() / max(-d.sum(), 1e-9)
    L += ['', '## 참고 (판정 없음)', '',
          f'- 대표의 ABAC 대비 비용 이득 가운데 {share * 100:.0f}% 가 이득이 가장 큰 {DROP}일에서 나온다',
          '- 지금까지 쓴 묶음 단위 구간 (묶음 5개의 t 구간, 날짜를 고정) 과 비교: '
          + '; '.join(f"{names[b]} {ci(np.array([per_day(g, 'cost')[1].mean() for g in R]) - series(gs, 'cost')[1].mean(), '{:+.3f}')}" for b, gs in base.items() if len(gs) == 1),
          '', '## 날짜별 하루 비용 (구성원 평균)', '', '| 날짜 | 대표 | ABAC | 차단만 | 대표 빼기 ABAC |', '| --- | --- | --- | --- | --- |']
    _, ac = series(base['abac'], 'cost')
    _, bc = series(base['brake'], 'cost')
    for i, dd in enumerate(days):
        L.append(f'| {dd} | {rc[i]:.1f} | {ac[i]:.1f} | {bc[i]:.1f} | {rc[i] - ac[i]:+.1f} |')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report')
    for k in ('e44', 'e45', 'md'):
        r.add_argument(f'--{k}', required=True)
    report(p.parse_args())


if __name__ == '__main__':
    main()
