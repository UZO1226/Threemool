"""DARROW 비학습 기준 제어기 (Park 작성, Park/e43).

ABAC: 암모니아 기반 폭기 제어. NH4 피드백 PI 를 총 공기에 바로 거는 속도형 (Yoo/e68 baselines.py 와 같은 식,
Park 환경에 맞게 다시 씀). 정책과 같은 관측 (obs 의 NH4) 만 읽고 대리 모델을 쓰지 않는다.

    공기_t = clip(공기_{t-1} + Kp (e_t - e_{t-1}) + Ki e_t, 0, 9000),  e_t = NH4_t - 설정값

에피소드 첫 스텝의 공기_{t-1} 은 그날 기록의 첫 공기량, e_{t-1} 은 e_0 이다.
"""
import numpy as np

from darrow_env import AIR_MAX, air_to_action


class ABAC:
    def __init__(self, env, sp, kp, ki):
        self.env, self.sp, self.kp, self.ki = env, float(sp), float(kp), float(ki)
        self.air = self.e = None

    def __call__(self, obs):
        e = np.asarray(obs[:, 0], dtype=float) - self.sp
        if self.env.t == 0 or self.air is None or len(self.air) != len(e):
            self.air, self.e = np.asarray(self.env.x['air'][:, 0], dtype=float).copy(), e
        self.air = np.clip(self.air + self.kp * (e - self.e) + self.ki * e, 0.0, AIR_MAX)
        self.e = e
        return air_to_action(self.air)
