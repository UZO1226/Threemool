"""Park/e71 대표와 차단 위 예측 제어의 비용 차이 나누기 (Park 작성). 규칙은 README.md 를 따른다.
이미 있는 스텝 기록 (대표: Park/e44 묶음 5개, 차단 위 예측 제어와 다시 고른 ABAC: Park/e64) 으로 하루 비용 차이를 NH4 위반과 DO 위반,
차단이 개입한 스텝과 개입하지 않은 스텝, 하루 중 시간대로 나눈다. 새 평가는 없다.

    python run_e71.py report --e44 <결과>/e44 --e64 <결과>/e64 --md results.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e49'))
from run_e38 import COMBOS, key  # noqa: E402
from run_e49 import NBOOT, SEED, block_ci  # noqa: E402

BANDS = ((0, 24), (24, 48), (48, 72), (72, 95))   # 스텝 구간 (15분 스텝, 6시간씩, 마지막은 95 스텝까지)


def per_day(L, f):
    """날짜마다 구성원 평균한 값 (날짜 오름차순)."""
    days = np.unique(L['day'])
    v = f(L)
    return days, np.array([v[L['day'] == d].mean() for d in days])


def parts(L):
    """에피소드마다 (NH4 위반 스텝, DO 위반 스텝, 개입 스텝의 위반, 비개입 스텝의 위반, 시간대별 위반) 합."""
    c = L['v_nh4'] + L['v_do']
    u = L['used'] > 0
    out = {'nh4': L['v_nh4'].sum(1), 'do': L['v_do'].sum(1), 'cost': c.sum(1), 'used_cost': (c * u).sum(1), 'free_cost': (c * ~u).sum(1),
           'air': L['air'].mean(1)}
    for i, (lo, hi) in enumerate(BANDS):
        out[f'band{i}'] = c[:, lo:hi].sum(1)
    return out


def series(logs, name):
    vals = []
    for L in logs:
        p = parts(L)
        days, s = per_day(L, lambda _L: p[name])
        vals.append(s)
    return days, np.mean(vals, axis=0)


def report(a):
    load = lambda p: dict(np.load(p))  # noqa: E731
    R = [load(Path(a.e44) / f'log_{key(c)}.npz') for c in COMBOS]
    M = [load(Path(a.e64) / 'log_mpc_brake60.npz')]
    A = [load(Path(a.e64) / 'log_abac54.npz')]
    names = ['cost', 'nh4', 'do', 'used_cost', 'free_cost'] + [f'band{i}' for i in range(len(BANDS))]
    lab = {'cost': '하루 비용', 'nh4': 'NH4 위반', 'do': 'DO 위반', 'used_cost': '차단 개입 스텝의 위반', 'free_cost': '차단 비개입 스텝의 위반'}
    for i, (lo, hi) in enumerate(BANDS):
        lab[f'band{i}'] = f'{lo * 15 // 60:02d}:00 ~ {hi * 15 // 60:02d}:00 위반' if hi < 95 else f'{lo * 15 // 60:02d}:00 ~ 23:45 위반'
    rng = np.random.default_rng(SEED)
    L = [f'검증 40일. 날짜마다 구성원 5개 평균 (대표는 묶음 5개도 평균). 위반 스텝 수는 하루 비용과 같은 단위. 블록 구간은 Park/e49 와 같은 이어진 5일 원형 이동 블록 {NBOOT:,}번 (seed {SEED}).', '']
    D = {}
    for other, nm in ((M, '차단 위 예측 제어 (Park/e60)'), (A, '다시 고른 ABAC (Park/e54)')):
        L += [f'### 대표 빼기 {nm}', '', '| 항목 | 대표 | 비교 대상 | 차이 | 블록 95% 구간 | 비용 차이 중 몫 |', '| --- | --- | --- | --- | --- | --- |']
        _, tot_r = series(R, 'cost')
        _, tot_o = series(other, 'cost')
        tot = (tot_r - tot_o).mean()
        for n in names:
            _, r = series(R, n)
            _, o = series(other, n)
            d = r - o
            lo, hi = block_ci(d, rng)
            D[(nm, n)] = d.mean()
            L.append(f'| {lab[n]} | {r.mean():.3f} | {o.mean():.3f} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {d.mean() / tot:.0%} |')
        L.append('')
    m = '차단 위 예측 제어 (Park/e60)'
    sh = D[(m, 'do')] / D[(m, 'cost')]
    sf = D[(m, 'free_cost')] / D[(m, 'cost')]
    L += ['## 판정', '',
          f"- H1 (대표 빼기 차단 위 예측 제어의 비용 차이 중 DO 위반 몫이 절반 이상인가): DO 몫 {sh:.0%}: {'지지' if sh >= 0.5 else '지지 안 됨'}",
          f"- H2 (그 비용 차이 중 차단 비개입 스텝 몫이 절반 이상인가): 비개입 몫 {sf:.0%}: {'지지' if sf >= 0.5 else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report')
    for k in ('e44', 'e64', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    report(a)


if __name__ == '__main__':
    main()
