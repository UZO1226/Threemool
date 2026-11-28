"""Park/e92 차단의 회복 계획의 몫, 학습 구간 188일 날짜 블록 (Park 작성). 규칙은 README.md 를 따른다.
차단 위 ABAC (설정값 3.2) 에서 대표 차단의 회복 계획만 끈 것을 학습 구간 188일에서 돌리고, Park/e86 의 대표 차단 쪽 (abac_b) 과 날짜마다 짝지어
하루 비용과 공기량 차이의 날짜 블록 95% 구간을 낸다.

    python run_e92.py run --part <i> --nparts 4 --data .. --surrogate .. --out <결과>/e92
    python run_e92.py report --out <결과>/e92 --e86 <결과>/e86 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e86'))
from run_e86 import BLOCK_NOTE, KI, KP, SP, ABAC, brake, ci, daily, load_log, run_chunks, three  # noqa: E402


def make(env, dyn, n):
    return ABAC(env, SP, KP, KI), brake(dyn, plan=None)


def pair_table(base, new, label_new, label_base, unit):
    """(new 빼기 base) 의 비용, 공기량 블록 구간 표 줄과 값 (Park/e93 도 씀)."""
    d0, c0, a0 = daily([base])
    d1, c1, a1 = daily([new])
    assert d0 == d1
    dc, da = c1 - c0, a1 - a0
    lo, hi = ci(dc)
    alo, ahi = ci(da)
    rows = ['| 방법 | 공기량 | 하루 비용 |', '| --- | --- | --- |', f'| {label_base} | {a0.mean():,.0f} | {c0.mean():.3f} |', f'| {label_new} | {a1.mean():,.0f} | {c1.mean():.3f} |', '',
            f'| 차이 ({unit}) | 평균 | 블록 95% 구간 | 나은 날 / 같은 날 / 나쁜 날 |', '| --- | --- | --- | --- |',
            f'| 하루 비용 | {dc.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {(dc < 0).sum()} / {(dc == 0).sum()} / {(dc > 0).sum()} |',
            f'| 하루 공기량 | {da.mean():+,.0f} ({da.mean() / a0.mean() * 100:+.1f}%) | {alo:+,.0f} ~ {ahi:+,.0f} | {(da < 0).sum()} / {(da == 0).sum()} / {(da > 0).sum()} |']
    return rows, (dc.mean(), lo, hi), (da.mean(), alo, ahi)


def report(a):
    rows, c, a_ = pair_table(load_log(a.out, 'noplan'), load_log(a.e86, 'abac_b'), '대표 차단 (회복 계획 있음, Park/e86)', '회복 계획 없는 차단', '회복 계획 있음 빼기 없음')
    L = [f'학습 구간 188일 x 구성원 5개. 차단 위 ABAC (설정값 {SP}, Kp {KP:,.0f}, Ki {KI:,.0f}). 회복 계획만 끔 (`plan=None`), 나머지 차단 설정은 같음. {BLOCK_NOTE}', ''] + rows
    L += ['', '## 판정', '',
          f'- H1 (회복 계획이 하루 비용을 낮추는가, 블록 구간이 0 아래): 차이 {c[0]:+.3f}, 구간 {c[1]:+.3f} ~ {c[2]:+.3f}: {three(c[1], c[2])}',
          f'- 참고 (판정 없음, 공기량): 차이 {a_[0]:+,.0f}, 구간 {a_[1]:+,.0f} ~ {a_[2]:+,.0f}: {three(a_[1], a_[2], "회복 계획이 적음", "회복 계획이 많음")}']
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e86', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_chunks(a, [('noplan', 'train')], make) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
