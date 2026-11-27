"""Park/e91 공식 기준선 PPO-Lag 에 차단만 씌웠을 때, 날짜 블록 (Park 작성). 규칙은 README.md 를 따른다.
PPO-Lag (seed 0 ~ 4) 에 대표 차단을 씌운 것과 씌우지 않은 것을 날짜마다 짝지어 하루 비용과 공기량 차이의 날짜 블록 95% 구간을 낸다.
검증 40일의 차단 없는 쪽은 Park/e89 기록을 쓰고, 나머지 (검증 차단, 학습 구간 188일 둘 다) 는 새로 돈다.

    python run_e91.py run --part <i> --nparts 8 --data .. --surrogate .. --runs <runs> --out <결과>/e91
    python run_e91.py report --out <결과>/e91 --e89 <결과>/e89 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e89'))
from run_e89 import BLOCK_NOTE, SEEDS, ci, daily, load_log, run_chunks, three  # noqa: E402
from run_e86 import brake  # noqa: E402
from run_e38 import load  # noqa: E402
from train_darrow import agent_act  # noqa: E402

JOBS = [(f'vppob{s}', 'val') for s in SEEDS] + [(f'ppo{s}', 'train') for s in SEEDS] + [(f'ppob{s}', 'train') for s in SEEDS]


def maker(runs):
    def make(env, dyn, n):
        s = int(n.lstrip('vpob'))
        g = load(runs, 'ppo', s)
        return (lambda obs: agent_act(g, obs)), (brake(dyn) if 'ppob' in n else None)
    return make


def report(a):
    L = [f'PPO-Lag (seed 0 ~ 4) 에 대표 차단을 씌운 것 빼기 씌우지 않은 것. 날짜마다 구성원과 seed 평균. {BLOCK_NOTE}', '',
         '| 날 | 날 수 | 차단 없음 공기량 / 비용 | 차단 공기량 / 비용 | 공기량 차이 (블록 95% 구간) | 하루 비용 차이 | 블록 95% 구간 | 차단이 나은 날 / 같은 날 / 나쁜 날 | 판정 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    R = {}
    for w, nb_dir, bp, label in (('val', a.e89, 'vppob', '검증'), ('train', a.out, 'ppob', '학습 구간 전체')):
        nb = [load_log(nb_dir, f'ppo{s}') for s in SEEDS]
        b = [load_log(a.out, f'{bp}{s}') for s in SEEDS]
        d0, c0, a0 = daily(nb)
        d1, c1, a1 = daily(b)
        assert d0 == d1
        dc, da = c1 - c0, a1 - a0
        lo, hi = ci(dc)
        alo, ahi = ci(da)
        R[w] = (dc.mean(), lo, hi, da.mean(), alo, ahi)
        L.append(f'| {label} | {len(d0)} | {a0.mean():,.0f} / {c0.mean():.3f} | {a1.mean():,.0f} / {c1.mean():.3f} | {da.mean():+,.0f} ({alo:+,.0f} ~ {ahi:+,.0f}) | '
                 f'{dc.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {(dc < 0).sum()} / {(dc == 0).sum()} / {(dc > 0).sum()} | {three(lo, hi)} |')
    L += ['', '## 판정', '']
    for tag, w, what in (('H1', 'val', '검증 40일'), ('H2', 'train', '학습 구간 188일')):
        m, lo, hi = R[w][:3]
        L.append(f'- {tag} ({what} 에서 차단이 PPO-Lag 의 하루 비용을 낮추는가, 블록 구간이 0 아래): 차이 {m:+.3f}, 구간 {lo:+.3f} ~ {hi:+.3f}: {three(lo, hi)}')
    for w, what in (('val', '검증 40일'), ('train', '학습 구간 188일')):
        m, lo, hi = R[w][3:]
        L.append(f'- 참고 (판정 없음, {what} 공기량): 차이 {m:+,.0f}, 구간 {lo:+,.0f} ~ {hi:+,.0f}: {three(lo, hi, "차단이 적음", "차단이 많음")}')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=8)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e89', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_chunks(a, JOBS, maker(a.runs)) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
