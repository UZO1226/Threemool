"""폭기 제어 Safe RL 환경 (gymnasium)

- 한 에피소드는 실제 운전 하루(5분 x 287 스텝)다. 유입량, 수온, 공정 단계(호기, 무산소)는 그날 기록을 그대로 재생한다
- 행동은 폭기조 1번 공기량이다. [-1, 1] 을 [0, AIR_MAX] Nm3/h 로 바꾼다. 호기 단계가 아니면 공기를 넣지 않는다 (실제 운전과 같음)
- 상태 전이는 surrogate.py 의 앙상블 모델이 계산한다. 에피소드마다 앙상블 구성원 하나를 고른다
- 보상은 에너지 대용치인 공기량의 음수 (-air / AIR_MAX)
- 비용은 스텝마다 NH4 > NH4_MAX 이면 1, DO > DO_MAX 이면 1 을 더한다 (info["cost"])

Yoo/e03 확장 (기본값 0 이면 e02 와 같다):
- pess_k (κ): 비관적 비용. 앙상블 구성원 전체의 다음 상태 예측 표준편차 σ 를 구해
  "고른 구성원의 예측 + κσ" 가 한도를 넘으면 위반으로 센다. info 의 v_nh4, v_do 는 언제나 실제(κ=0) 위반이다
- unc_pen (λ_u): 보상에서 λ_u x (σ_NH4 / NH4_MAX + σ_DO / DO_MAX) 를 뺀다 (MOPO 방식 불확실성 벌점)
- track_unc: 학습에 쓰지 않더라도 σ 를 계산해 info["unc"] 에 남긴다 (평가용)
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from surrogate import Surrogate


def load_surrogate(path):
    """.json 은 학습용 MLP 앙상블, .pkl 은 평가 전용 GBM 앙상블 (surrogate_gbm.py)"""
    if str(path).endswith(".pkl"):
        from surrogate_gbm import GBMSurrogate
        return GBMSurrogate(path)
    return Surrogate(path)

AIR_MAX = 8000.0  # Nm3/h, 운전 기록 99% 분위수 6458 보다 조금 크게
NH4_MAX = 4.0  # mg N/L
DO_MAX = 2.0  # mg O2/L, 과포기 상한
STEP_H = 5 / 60


class AerationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_path, surrogate_path, split="train", seed=None, fixed_episodes=None,
                 pess_k=0.0, unc_pen=0.0, track_unc=False):
        """fixed_episodes: [(날짜, 구성원), ...] 을 주면 그 순서대로 돌고, 없으면 무작위로 고른다"""
        d = pd.read_pickle(data_path)
        self.days = d["days"][split]
        r = d["data"]
        self.day_data = {}
        for day in self.days:
            g = r.loc[day]
            h = (g.index.hour + g.index.minute / 60).to_numpy()
            ph = g.phase.to_numpy()
            since = np.zeros(len(ph))
            for t in range(1, len(ph)):
                since[t] = since[t - 1] + 1 if ph[t] == ph[t - 1] else 0
            self.day_data[day] = dict(
                nh4=g.nh4.to_numpy(), do=g.do.to_numpy(), air=g.air.to_numpy(), phase=ph,
                temp=g.temp.to_numpy(), q_in=g.q_in.to_numpy(), hour=h, since=since)
        self.sur = load_surrogate(surrogate_path)
        self.n_members = self.sur.n_members
        self.pess_k, self.unc_pen = float(pess_k), float(unc_pen)
        self.need_all = self.pess_k > 0 or self.unc_pen > 0 or track_unc
        self.rng = np.random.default_rng(seed)
        self.fixed = list(fixed_episodes) if fixed_episodes else None
        self._k = 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.T = len(next(iter(self.day_data.values()))["nh4"]) - 1

    def _obs(self):
        x, t = self.x, self.t
        ph = int(x["phase"][t])
        s, c = np.sin(2 * np.pi * x["hour"][t] / 24), np.cos(2 * np.pi * x["hour"][t] / 24)
        return np.array([
            self.nh4, self.do, ph == 0, ph == 1, ph == 2, x["temp"][t] / 10, x["q_in"][t] / 1000,
            s, c, min(x["since"][t], 24) / 12, self.prev_air / AIR_MAX,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed:
            self.day, self.member = self.fixed[self._k % len(self.fixed)]
            self._k += 1
        else:
            self.day = self.days[self.rng.integers(len(self.days))]
            self.member = int(self.rng.integers(self.n_members))
        self.x = self.day_data[self.day]
        self.t = 0
        self.nh4, self.do = float(self.x["nh4"][0]), float(self.x["do"][0])
        self.prev_air = 0.0
        return self._obs(), {"cost": 0.0}

    def step(self, action):
        x, t = self.x, self.t
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        air = (a + 1) / 2 * AIR_MAX if x["phase"][t] == 2 else 0.0
        args = (self.nh4, self.do, air, x["phase"][t], x["temp"][t], x["q_in"][t], x["hour"][t])
        sd_n = sd_d = 0.0
        if self.need_all:
            nh4_all, do_all = self.sur.step_all(*args)
            self.nh4, self.do = float(nh4_all[self.member]), float(do_all[self.member])
            sd_n, sd_d = float(nh4_all.std()), float(do_all.std())
        else:
            self.nh4, self.do = self.sur.step(*args, self.member)
        self.t += 1
        self.prev_air = air
        v_nh4, v_do = float(self.nh4 > NH4_MAX), float(self.do > DO_MAX)
        c_nh4 = float(self.nh4 + self.pess_k * sd_n > NH4_MAX)
        c_do = float(self.do + self.pess_k * sd_d > DO_MAX)
        unc = sd_n / NH4_MAX + sd_d / DO_MAX
        info = {"cost": c_nh4 + c_do, "v_nh4": v_nh4, "v_do": v_do, "air": air, "unc": unc, "terminate": False}
        trunc = self.t >= self.T
        return self._obs(), -air / AIR_MAX - self.unc_pen * unc, False, trunc, info


def historical_air(env: AerationEnv):
    """현재 에피소드 날짜의 실제 운전 공기량을 행동 값([-1, 1])으로 바꾼다"""
    return np.clip(env.x["air"] / AIR_MAX * 2 - 1, -1, 1)
