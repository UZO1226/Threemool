"""e68 환경: 하루 NH4 평균 비용, 남은 예산 관측, 새 시험 구간.

문헌 기준에 맞춰 비용을 다시 정의한다 (README 근거 표).
- 하루 비용: 그날 NH4 평균 (행동 뒤 상태 95개) 이 4 mg/L 를 넘으면 1, 아니면 0.
- 스텝 비용: 그날 NH4 누적합이 4 x 95 = 380 을 처음 넘는 스텝에서 1. 하루 합은 하루 비용과 정확히 같다.
- 관측: e30 의 11개 + 남은 예산 (380 - 누적합) / 380 (-1 ~ 1 로 자름) + 남은 스텝 비율. 누적합이 관측에 있어
  비용이 상태의 함수가 된다.
- DO 는 비용에서 뺀다. 보고만 한다.
- 행동: 총 공기 [-1, 1] -> 0 ~ 9,000, 배분은 그 시각 기록 배분. 구간 공기는 개발 학습 날짜의 구간별 99 백분위로
  자르고 실제 넣은 공기를 보상과 지표에 쓴다.
- 날짜: e63 준비물의 train 188일을 7일 블록으로 나눠 블록 3, 7, 11, 15, 19, 23 (42일) 을 새 시험 구간 holdout 으로,
  나머지 146일을 dev 학습으로 쓴다. 기존 test 는 쓰지 않는다. val 40일은 보고용이며 선택에 쓰지 않는다.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from darrow_env import AIR_MAX, DO_MAX, STEP_H, Dynamics, action_to_air, load_days  # noqa: E402,F401

PREPARED = Path('/home/jio/fcsrl_src/results/e63/prepared')
NH4_LIMIT = 4.0
STEPS = 95
BUDGET = NH4_LIMIT * STEPS
HOLDOUT_BLOCKS = (3, 7, 11, 15, 19, 23)
OBS_DIM = 13


def load(prepared=PREPARED):
    splits, days = load_days(prepared / 'data.pkl')
    train = sorted(splits['train'])
    assert len(train) == 188 and train == list(splits['train'])
    holdout = [d for i, d in enumerate(train) if i // 7 in HOLDOUT_BLOCKS]
    dev = [d for d in train if d not in set(holdout)]
    assert len(holdout) == 42 and len(dev) == 146
    out = dict(dev=dev, holdout=holdout, val=list(splits['val']))
    assert not set(out['val']) & set(train) and 'test' not in out
    return out, days


def caps(days, dev):
    rows = np.concatenate([days[d]['zone'] for d in dev])
    return np.percentile(rows, 99, axis=0)


class DailyEnv:
    """e30 BatchEnv 와 같은 전이 (같은 Dynamics.step 호출) 에 하루 예산 상태와 구간 상한을 더한다."""

    def __init__(self, day_data, dynamics, cap):
        self.day_data, self.dyn, self.cap = day_data, dynamics, np.asarray(cap, float)
        assert len(next(iter(day_data.values()))['state']) - 1 == STEPS
        self.T = STEPS

    def reset(self, episodes):
        self.episodes = list(episodes)
        X = [self.day_data[d] for d, _ in self.episodes]
        self.x = {k: np.stack([x[k] for x in X]) for k in ('state', 'exo', 'frac', 'air', 'zone')}
        self.member = np.array([m for _, m in self.episodes])
        self.t = 0
        self.state = self.x['state'][:, 0].copy()
        self.prev_air = self.x['air'][:, 0].copy()
        self.prev_zone = self.x['zone'][:, 0].copy()
        self.nh4_sum = np.zeros(len(self.episodes))
        return self.obs()

    def obs(self):
        exo = self.x['exo'][:, self.t]
        budget = np.clip((BUDGET - self.nh4_sum) / BUDGET, -1.0, 1.0)
        return np.column_stack([self.state, exo[:, 0] / 1000.0, exo[:, 1] / 10.0, exo[:, 2], exo[:, 3],
                                exo[:, 4] / 50.0, self.prev_air / AIR_MAX, budget,
                                np.full(len(budget), (STEPS - self.t) / STEPS)]).astype(np.float32)

    def zone_air(self, action):
        want = action_to_air(np.asarray(action, float).reshape(-1))[:, None] * self.x['frac'][:, self.t]
        return np.minimum(want, self.cap[None, :])

    def step(self, action):
        t = self.t
        zone = self.zone_air(action)
        air = zone.sum(1)
        self.state = self.dyn.step(self.state, self.x['exo'][:, t], zone, self.prev_zone, self.member)
        self.prev_air, self.prev_zone = air, zone
        before = self.nh4_sum
        self.nh4_sum = before + self.state[:, 0]
        cost = ((self.nh4_sum > BUDGET) & (before <= BUDGET)).astype(float)
        self.t += 1
        info = dict(air=air, zone=zone, v_nh4=(self.state[:, 0] > NH4_LIMIT).astype(float),
                    v_do=(self.state[:, 2:].max(1) > DO_MAX).astype(float))
        return self.obs(), -air / AIR_MAX, cost, self.t >= self.T, info


def episodes(dates, n_members=5):
    return [(d, m) for d in dates for m in range(n_members)]


def rollout(env, dates, act):
    """모든 (날짜, 구성원) 을 결정적으로 돈다. act(raw_obs) -> (N,) 행동 [-1, 1]."""
    eps = episodes(dates, env.dyn.n_members)
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
    return metrics(states, airs, costs, eps), states, airs, zones


def metrics(states, airs, costs, eps):
    nh4 = states[:, :, 0]
    day_mean = nh4.mean(1)
    day_viol = nh4.sum(1) > BUDGET
    assert np.array_equal(costs.sum(1), day_viol.astype(float)), '스텝 비용 합과 하루 비용이 다르다'
    do_max = states[:, :, 2:].max(2)
    air_day = (airs * STEP_H).sum(1)
    return dict(n_episodes=len(eps), air_m3=float(air_day.mean()),
                air_m3_ok_days=float(air_day[~day_viol].mean()) if (~day_viol).any() else None,
                air_m3_violated_days=float(air_day[day_viol].mean()) if day_viol.any() else None,
                day_violation_pct=float(100 * day_viol.mean()), nh4_mean=float(nh4.mean()),
                nh4_day_mean_p90=float(np.percentile(day_mean, 90)), nh4_max_day_mean=float(day_mean.max()),
                nh4_step_viol_pct=float(100 * (nh4 > NH4_LIMIT).mean()), nh4_excess_mg_h_L=float((np.maximum(nh4 - NH4_LIMIT, 0) * STEP_H).sum(1).mean()),
                do_step_viol_pct=float(100 * (do_max > DO_MAX).mean()), no3_mean=float(states[:, :, 1].mean()),
                worst_violated_day_excess=float(np.where(day_viol, day_mean - NH4_LIMIT, 0).max()))
