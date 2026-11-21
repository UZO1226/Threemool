"""e74 환경: e68 환경에서 비용만 바꾼다. 스텝 비용 = 1[그 스텝 NH4 > 4 또는 그날 누적합이 380 을 넘은 뒤].

e68 ~ e73 비용 (예산 초과 스텝) 은 하루 평균만 보므로 NH4 를 4 근처에서 오르내리게 해도 비용이 없었다. e73 에서 차단 아래
RL 은 한도를 지키면서도 NH4 > 4 스텝이 하루 약 27개 (기록 운전 재생 4.75, ABAC 2.38) 였다. 사용자 결정 (2026-11-21):
순간 초과 스텝을 비용에 넣고 한도는 dev 기록 운전 재생 수준 (하루 4.75 스텝) 으로 둔다.

관측, 보상, 행동, 날짜, 전이는 e68 과 같다 (남은 예산 관측도 그대로). 이 모듈을 import 하면 env68 의 DailyEnv 와 rollout 을
바꿔 끼운다. 지표는 e68 지표에 step_cost (하루 비용 스텝 수) 를 더한다. e68 지표의 over_budget_steps 는 누적합으로 다시 계산한다.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e68'))
import env68 as E  # noqa: E402

EVAL_LIMIT = 4.75   # dev 기록 운전 재생의 하루 비용 스텝 (calibrate74 로 확인)
_BaseEnv = E.DailyEnv
_base_rollout = E.rollout


def step_cost(nh4_after, nh4_sum_after):
    return ((nh4_after > E.NH4_LIMIT) | (nh4_sum_after > E.BUDGET)).astype(float)


class DailyEnv74(_BaseEnv):
    def step(self, action):
        obs, rew, _, done, info = super().step(action)
        return obs, rew, step_cost(self.state[:, 0], self.nh4_sum), done, info


def rollout74(env, dates, act):
    eps = E.episodes(dates, env.dyn.n_members)
    obs = env.reset(eps)
    states, airs, zones, costs = [], [], [], []
    for _ in range(env.T):
        a = np.asarray(act(obs), dtype=float).reshape(len(eps))
        obs, _, cost, done, info = env.step(a)
        states.append(env.state.copy())
        airs.append(info['air'])
        zones.append(info['zone'])
        costs.append(cost)
    assert done
    states, airs, zones, costs = (np.stack(x, 1) for x in (states, airs, zones, costs))
    nh4 = states[:, :, 0]
    assert np.array_equal(costs, step_cost(nh4, np.cumsum(nh4, 1)))
    res = E.metrics(states, airs, (np.cumsum(nh4, 1) > E.BUDGET).astype(float), eps)
    res['step_cost'] = float(costs.sum(1).mean())
    return res, states, airs, zones


E.DailyEnv = DailyEnv74
E.rollout = rollout74
