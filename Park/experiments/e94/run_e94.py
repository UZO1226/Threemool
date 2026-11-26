"""Park/e94 학습 구간 188일을 앞 반과 뒤 반으로 나눈 차단의 몫 일관성 (Park 작성). 규칙은 README.md 를 따른다.
새 평가 없이 Park/e86 (ABAC 에서 차단의 몫) 과 Park/e88 (예측 제어에서 차단의 몫) 의 스텝 기록을 날짜순 앞 94일, 뒤 94일로 나눠
각 반에서 날짜 블록 95% 구간을 낸다. e88 은 반마다 그 반의 전체 공기량으로 다시 보간한다.

    python run_e94.py report --e86 <결과>/e86 --e87 <결과>/e87 --e88 <결과>/e88 --md results.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e88'))
from run_e88 import LAMS  # noqa: E402
from run_e86 import BLOCK_NOTE, ci, daily, load_log, matched, three  # noqa: E402


def halves(days):
    n = len(days) // 2
    return {'앞 반': slice(0, n), '뒤 반': slice(n, len(days))}


def report(a):
    d0, c0, _ = daily([load_log(a.e86, 'abac_nb')])
    d1, c1, _ = daily([load_log(a.e86, 'abac_b')])
    assert d0 == d1
    dm, mc, ma = daily([load_log(a.e87, 'mpc12')])
    nb = []
    for lam in LAMS:
        d, c, ar = daily([load_log(a.e88, f'nb{lam:g}')])
        assert d == dm
        nb.append((ar, c, f'가중치 {lam:g}'))
    L = [f'학습 구간 188일을 날짜순 앞 {len(d0) // 2}일 ({d0[0]} ~ {d0[len(d0) // 2 - 1]}) 과 뒤 {len(d0) - len(d0) // 2}일 ({d0[len(d0) // 2]} ~ {d0[-1]}) 로 나눔. {BLOCK_NOTE}', '',
         '| 비교 | 반 | 하루 비용 차이 | 블록 95% 구간 | 판정 |', '| --- | --- | --- | --- | --- |']
    R = {}
    for half, sl in halves(d0).items():
        d = (c1 - c0)[sl]
        lo, hi = ci(d)
        R[('e86', half)] = (lo, hi)
        L.append(f'| ABAC 에서 차단의 몫 (Park/e86) | {half} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {three(lo, hi)} |')
    for half, sl in halves(dm).items():
        m, how = matched([(ar[sl].mean(), c[sl], s) for ar, c, s in nb], ma[sl].mean())
        if m is None:
            R[('e88', half)] = None
            L.append(f'| 예측 제어에서 차단의 몫 (Park/e88) | {half} | 범위 밖 | | 판정 안 함 |')
            continue
        d = mc[sl] - m
        lo, hi = ci(d)
        R[('e88', half)] = (lo, hi)
        L.append(f'| 예측 제어에서 차단의 몫 (Park/e88, {how}) | {half} | {d.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {three(lo, hi)} |')

    def both(k):
        rs = [R[(k, h)] for h in ('앞 반', '뒤 반')]
        if any(r is None for r in rs):
            return '판정 안 함'
        return '지지' if all(hi < 0 for _, hi in rs) else '지지 안 됨'
    L += ['', '## 판정', '',
          f"- H1 (ABAC 에서 차단의 몫이 앞 반과 뒤 반 모두에서 블록 구간 0 아래인가): {both('e86')}",
          f"- H2 (예측 제어에서 차단의 몫이 앞 반과 뒤 반 모두에서 블록 구간 0 아래인가): {both('e88')}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report')
    for k in ('e86', 'e87', 'e88', 'md'):
        r.add_argument(f'--{k}', required=True)
    report(p.parse_args())


if __name__ == '__main__':
    main()
