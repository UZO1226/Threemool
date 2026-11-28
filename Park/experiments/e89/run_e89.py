"""Park/e89 대표 대 공식 기준선 PPO-Lag, 검증 40일 날짜 블록으로 공기와 비용 함께 (Park 작성). 규칙은 README.md 를 따른다.
공식 표의 PPO-Lag (차단 없음, seed 0 ~ 4) 를 스텝 기록과 함께 돌리고, 대표 (Park/e44 스텝 기록) 와 날짜마다 짝지어
하루 비용 차이와 하루 공기량 차이의 날짜 블록 95% 구간을 낸다. Park/e90 은 같은 비교를 학습 구간 마지막 40일에서 한다.

    python run_e89.py run --part <i> --nparts 5 --data .. --surrogate .. --runs <runs> --out <결과>/e89
    python run_e89.py report --out <결과>/e89 --e44 <결과>/e44 --md results.md
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e86'))
from run_e86 import BLOCK_NOTE, ci, daily, load_log, run_chunks, three  # noqa: E402
from run_e38 import BASE_PPO, COMBOS, key, load  # noqa: E402
from train_darrow import agent_act  # noqa: E402

SEEDS = range(5)


def jobs(which):
    return [(f'ppo{s}', which) for s in SEEDS]


def maker(runs):
    def make(env, dyn, n):
        g = load(runs, 'ppo', int(n[3:]))
        return (lambda obs: agent_act(g, obs)), None
    return make


def compare(rep_logs, ppo_logs, title):
    """대표 (묶음 목록) 대 PPO-Lag (seed 목록), 날짜별 평균끼리 짝지어 (대표 빼기 PPO-Lag) 비용과 공기량의 블록 구간."""
    d0, rc, ra = daily(rep_logs)
    d1, pc, pa = daily(ppo_logs)
    assert d0 == d1
    dc, da = rc - pc, ra - pa
    lo, hi = ci(dc)
    alo, ahi = ci(da)
    L = [title, '',
         '| 방법 | 공기량 | 하루 비용 |', '| --- | --- | --- |',
         f'| 대표 (묶음 5개 평균) | {ra.mean():,.0f} | {rc.mean():.3f} |',
         f'| PPO-Lag (차단 없음, seed 0 ~ 4 평균) | {pa.mean():,.0f} | {pc.mean():.3f} |', '',
         '| 차이 (대표 빼기 PPO-Lag) | 평균 | 블록 95% 구간 | 대표가 나은 날 / 같은 날 / 나쁜 날 | 판정 |', '| --- | --- | --- | --- | --- |',
         f'| 하루 비용 | {dc.mean():+.3f} | {lo:+.3f} ~ {hi:+.3f} | {(dc < 0).sum()} / {(dc == 0).sum()} / {(dc > 0).sum()} | {three(lo, hi, "대표가 낮음", "PPO-Lag 가 낮음")} |',
         f'| 하루 공기량 | {da.mean():+,.0f} ({da.mean() / pa.mean() * 100:+.1f}%) | {alo:+,.0f} ~ {ahi:+,.0f} | {(da < 0).sum()} / {(da == 0).sum()} / {(da > 0).sum()} | {three(alo, ahi, "대표가 적음", "PPO-Lag 가 적음")} |']
    return L, (dc.mean(), lo, hi), (da.mean(), alo, ahi), (pa.mean(), pc.mean())


def verdicts(c, a_, where):
    return ['', '## 판정', '',
            f'- H1 ({where} 에서 대표의 하루 비용이 PPO-Lag 보다 낮은가, 블록 구간이 0 아래): 차이 {c[0]:+.3f}, 구간 {c[1]:+.3f} ~ {c[2]:+.3f}: {three(c[1], c[2])}',
            f'- H2 ({where} 에서 대표의 하루 공기량이 PPO-Lag 보다 적은가, 블록 구간이 0 아래): 차이 {a_[0]:+,.0f}, 구간 {a_[1]:+,.0f} ~ {a_[2]:+,.0f}: {three(a_[1], a_[2])}',
            f'- 두 가설이 모두 지지면 "공기와 비용을 함께 줄임" 이 날짜 블록으로 확인된 것이다: {"예" if c[2] < 0 and a_[2] < 0 else "아니오"}']


def report(a):
    rep = [load_log(a.e44, key(c)) for c in COMBOS]
    ppo = [load_log(a.out, f'ppo{s}') for s in SEEDS]
    L, c, a_, (pa, pc) = compare(rep, ppo, f'검증 40일 x 구성원 5개. 날짜마다 구성원 평균 (대표는 묶음 5개, PPO-Lag 는 seed 5개도 평균). {BLOCK_NOTE}')
    L.insert(1, f'다시 돈 PPO-Lag 평균 {pa:,.0f} / {pc:.3f} (공식 표 {BASE_PPO["air_m3"]:,.0f} / {BASE_PPO["cost"]:.2f}).')
    L += verdicts(c, a_, '검증 40일')
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
    for k in ('out', 'e44', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_chunks(a, jobs('val'), maker(a.runs)) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
