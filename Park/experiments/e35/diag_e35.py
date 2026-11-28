"""Park/e35 같은 경로 재생 진단 (Park 작성). 학습 구간 마지막 40일, seed 0 ~ 3 앙상블, 원판 회복 계획으로 운전하면서
계획을 쓰는 스텝마다 원판 (OR 비용) 과 수정판 (합 비용) 이 고르는 공기량을 함께 계산한다. 실행은 원판 공기량으로 한다.

    python diag_e35.py --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --md diag.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import AIR_MAX, BatchEnv, Dynamics, all_episodes, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]


class TwinPlan(MinAirBrake):
    """원판으로 운전하면서 같은 상태에서 수정판의 선택도 기록한다."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.log = []   # (원판 공기량, 수정판 공기량) 계획 스텝마다

    def _plan_recovery(self, obs, env, air):
        nh_max, _ = self._rollout(obs, env, np.full((len(obs), 1), AIR_MAX), 1)
        rows = np.flatnonzero(nh_max[:, 0] > self.plan_trigger)
        self.plan_sum = True
        fixed = super()._plan_recovery(obs, env, air)
        self.plan_sum = False
        orig = super()._plan_recovery(obs, env, air)
        self.log += list(zip(orig[rows], fixed[rows]))
        return orig


def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'surrogate', 'runs', 'md'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)

    def load(s):
        ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
        ag.load(Path(a.runs) / 'dw_u16_runs' / f'A_s{s}' / 'model_final')
        return ag
    pol = EnsemblePolicy([lambda o, g=load(s): agent_act(g, o) for s in range(4)])
    br = TwinPlan(dyn, 3.5, 1, 'others', trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))
    obs = env.reset(all_episodes(splits['train'][-40:], dyn.n_members))
    N, done = len(obs), False
    while not done:
        act, _ = br(obs, np.asarray(pol(obs), dtype=float).reshape(N), env)
        obs, _, _, done, _ = env.step(act)
    lg = np.array(br.log)
    # 각 계획 스텝은 원판, 수정판 두 번 계산하므로 칸 수를 반으로 나눈다
    both, cells = br.plan_both / 2, br.plan_cells / 2
    diff = lg[:, 1] - lg[:, 0]
    ch = np.abs(diff) > 1e-9
    L = ['학습 구간 마지막 40일 x 구성원 5개 = 200 에피소드, seed 0 ~ 3 앙상블, 원판 회복 계획 경로.', '',
         '| 항목 | 값 |', '| --- | --- |',
         f'| 계획을 쓴 스텝 | {len(lg)} ({len(lg) / N:.2f} 회/일) |',
         f'| 후보 x 4 스텝 예측 칸 중 NH4 와 DO 위반이 함께 예측된 칸 | {both:.0f} / {cells:.0f} ({both / cells * 100:.2f}%) |',
         f'| 수정판이 다른 공기량을 고른 스텝 | {ch.sum()} ({ch.mean() * 100:.1f}%) |',
         f'| 그중 수정판이 공기를 줄인 스텝, 늘린 스텝 | {(diff < -1e-9).sum()}, {(diff > 1e-9).sum()} |',
         f'| 다른 스텝의 공기량 차이 평균 (Nm3/h) | {diff[ch].mean() if ch.any() else 0.0:+.0f} |',
         f'| 원판 선택이 9,000 인 스텝, 수정판 | {(lg[:, 0] >= 9000 - 1e-9).mean() * 100:.1f}%, {(lg[:, 1] >= 9000 - 1e-9).mean() * 100:.1f}% |']
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
