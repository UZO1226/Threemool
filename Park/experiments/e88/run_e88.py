"""Park/e88 예측 제어에서 차단의 몫, 학습 구간 188일 날짜 블록 (Park 작성). 규칙은 README.md 를 따른다.
차단 없는 예측 제어 (NH4 여유 1.2, 공기 가중치 0.7, 1.0, 1.4, 2.0) 를 학습 구간 188일에서 돌리고,
차단 위 예측 제어 기본 (Park/e87 의 mpc12) 의 공기량에서 차단 없는 쪽의 날짜별 비용을 보간해 날짜마다 짝지어 블록 95% 구간을 낸다.

    python run_e88.py run --part <i> --nparts 8 --data .. --surrogate .. --out <결과>/e88
    python run_e88.py report --out <결과>/e88 --e87 <결과>/e87 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e86'))
from run_e86 import BLOCK_NOTE, ci, daily, load_log, matched, run_chunks, three  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402

LAMS = (0.7, 1.0, 1.4, 2.0)
JOBS = [(f'nb{lam:g}', 'train') for lam in LAMS]


def make(env, dyn, n):
    return ShortMPC(env, dyn, H=4, margin=1.2, lam=float(n[2:]), objective='excess'), None


def report(a):
    days, bc, ba = daily([load_log(a.e87, 'mpc12')])
    pts = []
    for lam in LAMS:
        d, c, ar = daily([load_log(a.out, f'nb{lam:g}')])
        assert d == days
        pts.append((ar.mean(), c, f'가중치 {lam:g}'))
    m, how = matched(pts, ba.mean())
    L = [f'학습 구간 188일 x 구성원 5개. 예측 제어 (NH4 여유 1.2, 1시간). 차단 없는 쪽의 날짜별 비용을 차단 위 예측 제어 기본 (가중치 1.5) 의 전체 공기량에서 가중치 사이 선형 보간. {BLOCK_NOTE}', '',
         '- 차단 없는 예측 제어: ' + ', '.join(f'{s}: {a_:,.0f} / {c.mean():.3f}' for a_, c, s in sorted(pts, key=lambda p: p[0])),
         f'- 차단 위 예측 제어 기본 (Park/e87): {ba.mean():,.0f} / {bc.mean():.3f}', '']
    if m is None:
        L += ['## 판정', '', '- H1: 판정 안 함 (차단 위 공기량이 차단 없는 점들의 범위 밖)']
    else:
        d = bc - m
        lo, hi = ci(d)
        L += ['| 비교 | 보간 | 하루 비용 차이 (차단 빼기 차단 없음) | 블록 95% 구간 | 차단이 나은 날 / 같은 날 / 나쁜 날 | 판정 |', '| --- | --- | --- | --- | --- | --- |',
              f'| 차단 위 예측 제어 대 같은 공기량의 차단 없는 예측 제어 | {how} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()} | {three(lo, hi)} |',
              '', '## 판정', '', f'- H1 (학습 구간 188일에서 차단이 같은 공기량의 예측 제어 비용을 낮추는가, 블록 구간이 0 아래): 차이 {d.mean():+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: {three(lo, hi)}']
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
    for k in ('out', 'e87', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_chunks(a, JOBS, make) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
