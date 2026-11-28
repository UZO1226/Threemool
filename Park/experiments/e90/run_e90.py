"""Park/e90 대표 대 공식 기준선 PPO-Lag, 학습 구간 마지막 40일 날짜 블록으로 공기와 비용 함께 (Park 작성). 규칙은 README.md 를 따른다.
Park/e89 와 같은 비교를 학습 구간 마지막 40일에서 한다. 대표 스텝 기록은 Park/e83 을 쓰고, PPO-Lag (seed 0 ~ 4, 차단 없음) 는 새로 돈다.

    python run_e90.py run --part <i> --nparts 5 --data .. --surrogate .. --runs <runs> --out <결과>/e90
    python run_e90.py report --out <결과>/e90 --e83 <결과>/e83 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e89'))
from run_e89 import BLOCK_NOTE, COMBOS, SEEDS, compare, jobs, key, load_log, maker, run_chunks, verdicts  # noqa: E402


def report(a):
    rep = [load_log(a.e83, f'rep_{key(c)}') for c in COMBOS]
    ppo = [load_log(a.out, f'ppo{s}') for s in SEEDS]
    L, c, a_, _ = compare(rep, ppo, f'학습 구간 마지막 40일 x 구성원 5개. 날짜마다 구성원 평균 (대표는 묶음 5개, PPO-Lag 는 seed 5개도 평균). {BLOCK_NOTE}')
    L += verdicts(c, a_, '학습 구간 마지막 40일')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=5)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e83', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_chunks(a, jobs('trainlast'), maker(a.runs)) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
