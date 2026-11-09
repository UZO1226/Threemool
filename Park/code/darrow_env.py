"""Kaggle DARROW (팀 공식 데이터) 폭기 제어 환경, 배열 버전 (Park 작성, Park/e08).

팀 결과와 비교할 수 있도록 Lee/code/darrow/env_dw.py 와 같은 규칙을 새로 작성했다 (SOURCE.md).
상태 전이는 Lee 의 방법(Lee/code/darrow/surrogate_dw.py, 단조 제약 GBM 앙상블 5개) 으로 만든 대리 모델
pickle 을 읽어 쓴다. 이 모델은 공기량이 늘면 NH4 변화량이 줄고(비증가), 각 구간 DO 는 늘도록(비감소) 강제돼 있다.

- 에피소드: 실제 운전 하루 (15분 x 96 행, 전이 95개). 유입량, 수온, 시각, 유입 성분은 그날 기록을 재생한다
- 행동: 총 공기량 [-1, 1] -> [0, 9000] Nm3/h. 5개 구간 배분은 그 시각 기록된 운전의 비율
  (기록 총량이 0 이면 학습 구간 중앙값 비율)
- 상태: NH4, NO3, DO 1 ~ 3 구간. 관측 11개는 상태 5개, 유입량, 수온, 시각 sin, cos, 유입 NH4, 직전 공기량
- 보상: -공기량 / 9000. 비용: NH4 > 4 이면 1, 구간 DO 최댓값 > 2 이면 1 (스텝당 0, 1, 2). 하루 한도 12
"""
import pickle

import numpy as np
import pandas as pd

AIR_MAX = 9000.0
NH4_MAX = 4.0
DO_MAX = 2.0
STEP_H = 0.25
COST_LIMIT = 12.0
STATE = ['nh4', 'no3', 'do1', 'do2', 'do3']
EXO = ['q_in', 'temp', 'sin', 'cos', 'inf_s_nh', 'inf_s_nd', 'inf_x_nd', 'inf_s_s', 'inf_x_s']
ZONES = ['air1', 'air2', 'air3', 'air4', 'air5']


def action_to_air(action):
    return (np.clip(np.asarray(action, dtype=float), -1, 1) + 1) / 2 * AIR_MAX


def air_to_action(air):
    return np.asarray(air, dtype=float) / AIR_MAX * 2 - 1


class Dynamics:
    """대리 모델 pickle ({'models': [[GBM x 5 상태] x 5 구성원], 'hi': 상한}) 을 읽어 한 스텝 전이를 계산한다."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            z = pickle.load(f)
        self.models, self.hi = z['models'], np.asarray(z['hi'])
        self.n_members = len(self.models)

    def step(self, state, exo, zone, zone_prev, member):
        x = np.column_stack([state, exo, zone / 1000.0, zone_prev / 1000.0])
        delta = np.zeros_like(state)
        for m in np.unique(member):
            rows = member == m
            for j in range(len(STATE)):
                delta[rows, j] = self.models[int(m)][j].predict(x[rows])
        return np.clip(state + delta, 0.0, self.hi)


def load_days(data_path):
    d = pd.read_pickle(data_path)
    median = np.array([d['zone_fractions'][z] for z in ZONES])
    out = {}
    for split, days in d['days'].items():
        for day in days:
            g = d['data'].loc[day]
            z = g[ZONES].to_numpy(float)
            tot = z.sum(axis=1, keepdims=True)
            out[day] = {'state': g[STATE].to_numpy(float), 'exo': g[EXO].to_numpy(float),
                        'frac': np.where(tot > 1.0, z / np.maximum(tot, 1.0), median[None, :]),
                        'air': g.air.to_numpy(float), 'zone': z}
    return {k: list(v) for k, v in d['days'].items()}, out


class BatchEnv:
    def __init__(self, day_data, dynamics):
        self.day_data, self.dyn = day_data, dynamics
        self.T = len(next(iter(day_data.values()))['state']) - 1    # 95

    def reset(self, episodes):
        self.episodes = list(episodes)
        X = [self.day_data[d] for d, _ in self.episodes]
        self.x = {k: np.stack([x[k] for x in X]) for k in X[0]}
        self.member = np.array([m for _, m in self.episodes])
        self.t = 0
        self.state = self.x['state'][:, 0].copy()
        self.prev_air = self.x['air'][:, 0].copy()
        self.prev_zone = self.x['zone'][:, 0].copy()
        return self.obs()

    def obs(self):
        exo = self.x['exo'][:, self.t]
        return np.column_stack([self.state, exo[:, 0] / 1000.0, exo[:, 1] / 10.0, exo[:, 2], exo[:, 3],
                                exo[:, 4] / 50.0, self.prev_air / AIR_MAX]).astype(np.float32)

    def step(self, action):
        t = self.t
        air = action_to_air(np.asarray(action).reshape(-1))
        zone = air[:, None] * self.x['frac'][:, t]
        self.state = self.dyn.step(self.state, self.x['exo'][:, t], zone, self.prev_zone, self.member)
        self.prev_air, self.prev_zone = air, zone
        self.t += 1
        v_nh4 = (self.state[:, 0] > NH4_MAX).astype(float)
        v_do = (self.state[:, 2:].max(axis=1) > DO_MAX).astype(float)
        info = {'air': air, 'v_nh4': v_nh4, 'v_do': v_do}
        return self.obs(), -air / AIR_MAX, v_nh4 + v_do, self.t >= self.T, info

    def historical_action(self):
        return np.clip(air_to_action(self.x['air'][:, self.t]), -1, 1)


def all_episodes(days, n_members):
    return [(d, m) for d in days for m in range(n_members)]


def evaluate(env, days, act, shield=None):
    """모든 (날짜, 구성원) 을 한 번씩 돈다. 지표 정의는 Lee/code/darrow/env_dw.batched_rollout 과 같다."""
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    N = len(env.episodes)
    air = cost = vn = vd = np.zeros(N)
    used_steps = total_steps = 0.0
    done = False
    while not done:
        a = np.asarray(act(obs), dtype=float).reshape(N)
        if shield is not None:
            a, used = shield(obs, a, env)
            used_steps += float(used.sum())
            total_steps += N
        obs, _, c, done, info = env.step(a)
        air, cost = air + info['air'], cost + c
        vn, vd = vn + info['v_nh4'], vd + info['v_do']
    T = env.T
    out = {'air_m3': float((air * STEP_H).mean()), 'cost': float(cost.mean()),
           'nh4_viol_pct': float((100 * vn / T).mean()), 'do_viol_pct': float((100 * vd / T).mean()),
           'days_within_limit_pct': float(100 * np.mean(cost <= COST_LIMIT)), 'n_episodes': N}
    if shield is not None:
        out['intervention_pct'] = 100 * used_steps / max(total_steps, 1.0)
    return out
