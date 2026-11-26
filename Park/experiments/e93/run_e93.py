"""Park/e93 차단 NH4 여유를 줄였을 때의 공기와 비용, 학습 구간 188일 날짜 블록 (Park 작성). 규칙은 README.md 를 따른다.
차단 위 ABAC (설정값 3.2) 에서 대표 차단의 NH4 여유 표 모든 칸에서 0.2 를 뺀 것을 학습 구간 188일에서 돌리고, Park/e86 의 대표 차단 쪽 (abac_b) 과 날짜마다 짝짓는다.

    python run_e93.py run --part <i> --nparts 4 --data .. --surrogate .. --out <결과>/e93
    python run_e93.py report --out <결과>/e93 --e86 <결과>/e86 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e92'))
from run_e92 import BLOCK_NOTE, KI, KP, SP, ABAC, brake, load_log, pair_table, run_chunks, three  # noqa: E402
from run_e38 import BRAKE  # noqa: E402

DELTA = -0.2


def make(env, dyn, n):
    return ABAC(env, SP, KP, KI), brake(dyn, margins=[(u, m + DELTA) for u, m in BRAKE['margins']])


def report(a):
    rows, c, a_ = pair_table(load_log(a.e86, 'abac_b'), load_log(a.out, 'mneg02'), f'차단 NH4 여유 {DELTA:+.1f}', '대표 차단 (Park/e86)', f'여유 {DELTA:+.1f} 빼기 대표 차단')
    L = [f'학습 구간 188일 x 구성원 5개. 차단 위 ABAC (설정값 {SP}, Kp {KP:,.0f}, Ki {KI:,.0f}). 차단 NH4 여유 표 모든 칸에 {DELTA:+.1f}, 나머지 차단 설정은 같음. {BLOCK_NOTE}', ''] + rows
    L += ['', '## 판정', '',
          f'- H1 (여유를 줄이면 하루 비용이 높아지는가, 블록 구간이 0 위): 차이 {c[0]:+.3f}, 구간 {c[1]:+.3f} ~ {c[2]:+.3f}: {three(c[1], c[2], "반대 (낮아짐)", "지지")}',
          f'- H2 (여유를 줄이면 하루 공기량이 줄어드는가, 블록 구간이 0 아래): 차이 {a_[0]:+,.0f}, 구간 {a_[1]:+,.0f} ~ {a_[2]:+,.0f}: {three(a_[1], a_[2])}']
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
    run_chunks(a, [('mneg02', 'train')], make) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
