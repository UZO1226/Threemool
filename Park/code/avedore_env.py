"""Kaggle Avedøre 폭기 제어 환경, 여러 에피소드를 한꺼번에 굴리는 배열 버전 (Park 작성, Park/e07).

팀 결과와 비교할 수 있도록 Yoo/code/wwtp 의 AerationEnv 와 같은 규칙을 새로 작성했다 (SOURCE.md).
상태 전이는 Yoo 가 학습한 대리 모델 가중치 `Yoo/experiments/e02/surrogate.json` 을 그대로 읽어 쓴다.

- 에피소드: 실제 운전 하루 (5분 x 287 스텝). 유입량, 수온, 공정 단계는 그날 기록을 재생한다
- 행동: [-1, 1] 을 공기량 [0, 8000] Nm3/h 로 바꾼다. 호기 단계(phase 2) 가 아니면 공기량 0
- 상태 전이: 대리 모델 앙상블 구성원 하나 (에피소드마다 정한다). DO 를 먼저, 그 DO 로 NH4 를 예측한다
- 보상: -공기량 / 8000. 비용: NH4 > 4 이면 1, DO > 2 이면 1 을 더한다 (스텝당 0, 1, 2)
- 하루 끝은 시간 제한(truncation) 이다. 가치를 다음 상태로 이어 붙인다 (Yoo 와 같음)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AIR_MAX = 8000.0      # Nm3/h
NH4_MAX = 4.0         # mg N/L
DO_MAX = 2.0          # mg O2/L
STEP_H = 5 / 60
COST_LIMIT = 15.0     # 하루 비용 한도 (Yoo/e02 이후 팀 설정)
AIR_SCALE = Q_SCALE = 1000.0


def action_to_air(action, phase):
    a = np.clip(np.asarray(action, dtype=float), -1, 1)
    return np.where(np.asarray(phase) == 2, (a + 1) / 2 * AIR_MAX, 0.0)


def air_to_action(air):
    return np.asarray(air, dtype=float) / AIR_MAX * 2 - 1


class Ensemble:
    """저장된 가중치로 추론만 하는 ReLU MLP 앙상블 (입력 표준화, 출력 역표준화 포함)."""

    def __init__(self, p):
        self.mu, self.sd = np.array(p['x_mu']), np.array(p['x_sd'])
        self.ymu, self.ysd = p['y_mu'], p['y_sd']
        self.members = [[(np.array(W), np.array(b)) for W, b in m] for m in p['members']]

    def predict(self, x, member):
        h = (x - self.mu) / self.sd
        layers = self.members[member]
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0] * self.ysd + self.ymu


class Surrogate:
    def __init__(self, path):
        p = json.loads(Path(path).read_text())
        self.do, self.nh4 = Ensemble(p['do']), Ensemble(p['nh4'])
        self.n_members = len(self.do.members)

    def step(self, nh4, do, air, phase, temp, q_in, hour, member):
        """배열 입력. member 별로 나눠 다음 (NH4, DO) 를 낸다."""
        s, c = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        oh = np.stack([(phase == k).astype(float) for k in (0, 1, 2)], 1)
        do_n, nh4_n = np.empty(len(nh4)), np.empty(len(nh4))
        for m in np.unique(member):
            i = member == m
            xd = np.column_stack([do[i], air[i] / AIR_SCALE, oh[i], temp[i], nh4[i], q_in[i] / Q_SCALE, s[i], c[i]])
            do_n[i] = np.maximum(0.0, do[i] + self.do.predict(xd, m))
            xn = np.column_stack([nh4[i], do[i], do_n[i], oh[i], temp[i], q_in[i] / Q_SCALE, s[i], c[i]])
            nh4_n[i] = np.maximum(0.0, nh4[i] + self.nh4.predict(xn, m))
        return nh4_n, do_n


def load_days(data_path):
    """{split: [날짜]}, {날짜: 배열 dict}. Yoo/code/wwtp/prep.py 의 출력 pickle 을 읽는다."""
    d = pd.read_pickle(data_path)
    out = {}
    for split, days in d['days'].items():
        for day in days:
            g = d['data'].loc[day]
            ph = g.phase.to_numpy().round().astype(int)
            since = np.zeros(len(ph))
            for t in range(1, len(ph)):
                since[t] = since[t - 1] + 1 if ph[t] == ph[t - 1] else 0
            out[day] = dict(nh4=g.nh4.to_numpy(float), do=g.do.to_numpy(float), air=g.air.to_numpy(float),
                            phase=ph, temp=g.temp.to_numpy(float), q_in=g.q_in.to_numpy(float),
                            hour=(g.index.hour + g.index.minute / 60).to_numpy(float), since=since)
    return {k: list(v) for k, v in d['days'].items()}, out


class BatchEnv:
    """N 개 에피소드를 한꺼번에 굴린다. reset(episodes) 의 episodes 는 [(날짜, 구성원), ...]."""

    def __init__(self, day_data, surrogate):
        self.day_data, self.sur = day_data, surrogate
        self.T = len(next(iter(day_data.values()))['nh4']) - 1    # 287

    def reset(self, episodes):
        self.episodes = list(episodes)
        X = [self.day_data[d] for d, _ in self.episodes]
        self.x = {k: np.stack([x[k] for x in X]) for k in X[0]}
        self.member = np.array([m for _, m in self.episodes])
        self.t = 0
        self.nh4, self.do = self.x['nh4'][:, 0].copy(), self.x['do'][:, 0].copy()
        self.prev_air = np.zeros(len(self.episodes))
        return self.obs()

    def obs(self):
        t, x = self.t, self.x
        ph = x['phase'][:, t]
        s, c = np.sin(2 * np.pi * x['hour'][:, t] / 24), np.cos(2 * np.pi * x['hour'][:, t] / 24)
        return np.column_stack([self.nh4, self.do, ph == 0, ph == 1, ph == 2, x['temp'][:, t] / 10,
                                x['q_in'][:, t] / 1000, s, c, np.minimum(x['since'][:, t], 24) / 12,
                                self.prev_air / AIR_MAX]).astype(np.float32)

    def phase(self):
        return self.x['phase'][:, self.t]

    def step(self, action):
        t, x = self.t, self.x
        air = action_to_air(np.asarray(action).reshape(-1), x['phase'][:, t])
        self.nh4, self.do = self.sur.step(self.nh4, self.do, air, x['phase'][:, t], x['temp'][:, t],
                                          x['q_in'][:, t], x['hour'][:, t], self.member)
        self.t += 1
        self.prev_air = air
        v_nh4, v_do = (self.nh4 > NH4_MAX).astype(float), (self.do > DO_MAX).astype(float)
        info = {'air': air, 'v_nh4': v_nh4, 'v_do': v_do}
        return self.obs(), -air / AIR_MAX, v_nh4 + v_do, self.t >= self.T, info

    def historical_action(self):
        return np.clip(air_to_action(self.x['air'][:, self.t]), -1, 1)


def all_episodes(days, n_members):
    return [(d, m) for d in days for m in range(n_members)]


def evaluate(env, days, act, shield=None):
    """모든 (날짜, 구성원) 을 한 번씩 돈다. act(obs) -> (N,) 행동. shield(obs, action, env) -> (행동, 개입).

    반환: 에피소드 평균 지표 (Yoo/code/wwtp/evaluate.py 의 summarize_rows 와 같은 정의) 와 개입 비율.
    """
    obs = env.reset(all_episodes(days, env.sur.n_members))
    N = len(env.episodes)
    air = cost = vn = vd = np.zeros(N)
    used_steps = aerobic_steps = 0.0
    done = False
    while not done:
        a = np.asarray(act(obs), dtype=float).reshape(N)
        if shield is not None:
            a, used = shield(obs, a, env)
            aerobic = env.phase() == 2
            used_steps += float((used & aerobic).sum())
            aerobic_steps += float(aerobic.sum())
        obs, _, c, done, info = env.step(a)
        air, cost = air + info['air'] * STEP_H, cost + c
        vn, vd = vn + info['v_nh4'], vd + info['v_do']
    T = env.T
    out = {'air_m3': float(air.mean()), 'cost': float(cost.mean()),
           'nh4_viol_pct': float((100 * vn / T).mean()), 'do_viol_pct': float((100 * vd / T).mean()),
           'days_within_limit_pct': float(100 * np.mean(cost <= COST_LIMIT)), 'n_episodes': N}
    if shield is not None:
        out['intervention_pct'] = 100 * used_steps / max(aerobic_steps, 1.0)
    return out
