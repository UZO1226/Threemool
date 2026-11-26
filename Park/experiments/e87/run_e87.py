"""Park/e87 같은 차단 위의 예측 제어 대 ABAC, 학습 구간 188일 날짜 블록 (Park 작성). 규칙은 README.md 를 따른다.
차단 위 예측 제어 기본 (NH4 여유 1.2, 공기 가중치 1.5) 과 차단 위 ABAC (설정값 2.6, 2.9, 3.5, 그리고 Park/e86 의 3.2) 를 학습 구간 188일에서 돌리고,
예측 제어의 공기량에서 ABAC 의 날짜별 비용을 보간해 날짜마다 짝지어 블록 95% 구간을 낸다. 둘 다 학습하지 않는다.

    python run_e87.py run --part <i> --nparts 8 --data .. --surrogate .. --out <결과>/e87
    python run_e87.py report --out <결과>/e87 --e86 <결과>/e86 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e86'))
from run_e86 import BLOCK_NOTE, KI, KP, ABAC, brake, ci, daily, load_log, matched, run_chunks, three  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402

SPS = (2.6, 2.9, 3.5)     # 3.2 는 Park/e86 의 abac_b
JOBS = [('mpc12', 'train')] + [(f'ab{sp:g}', 'train') for sp in SPS]


def make(env, dyn, n):
    if n == 'mpc12':
        return ShortMPC(env, dyn, H=4, margin=1.2, lam=1.5, objective='excess'), brake(dyn)
    return ABAC(env, float(n[2:]), KP, KI), brake(dyn)


def report(a):
    days, mc, ma = daily([load_log(a.out, 'mpc12')])
    pts = []
    for name, out, sp in [(f'ab{s:g}', a.out, s) for s in SPS] + [('abac_b', a.e86, 3.2)]:
        d, c, ar = daily([load_log(out, name)])
        assert d == days
        pts.append((ar.mean(), c, f'설정값 {sp:g}'))
    m, how = matched(pts, ma.mean())
    L = [f'학습 구간 188일 x 구성원 5개. 둘 다 대표 차단 위. ABAC (Kp {KP:,.0f}, Ki {KI:,.0f}) 의 날짜별 비용을 예측 제어의 전체 공기량에서 설정값 사이 선형 보간. {BLOCK_NOTE}', '',
         '- 차단 위 ABAC: ' + ', '.join(f'{s}: {a_:,.0f} / {c.mean():.3f}' for a_, c, s in sorted(pts, key=lambda p: p[0])),
         f'- 차단 위 예측 제어 기본: {ma.mean():,.0f} / {mc.mean():.3f}', '']
    if m is None:
        L += ['## 판정', '', '- H1: 판정 안 함 (예측 제어 공기량이 ABAC 점들의 범위 밖)']
    else:
        d = mc - m
        lo, hi = ci(d)
        L += ['| 비교 | 보간 | 하루 비용 차이 (예측 제어 빼기 ABAC) | 블록 95% 구간 | 예측 제어가 나은 날 / 같은 날 / 나쁜 날 | 판정 |', '| --- | --- | --- | --- | --- | --- |',
              f'| 차단 위 예측 제어 기본 대 같은 공기량의 차단 위 ABAC | {how} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()} | {three(lo, hi)} |',
              '', '## 판정', '', f'- H1 (학습 구간 188일에서 예측 제어가 같은 공기량의 ABAC 보다 낮은가, 블록 구간이 0 아래): 차이 {d.mean():+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: {three(lo, hi)}']
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
    for k in ('out', 'e86', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_chunks(a, JOBS, make) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
