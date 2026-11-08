"""암모니아 기반 폭기 제어(ABAC) 비교 기준

방류 NH4 측정값(이상적 센서)으로 DO 설정값을 정하는 PI 제어다. 실무에서 쓰는 표준 형태
(NH4 가 목표보다 높으면 DO 설정값을 올리고, 낮으면 내린다)를 5분 간격으로 계산한다.

    sp = clip(sp_base + Kp (NH4 - NH4_ref) + I, 0.5, 3.0)
    I <- I + Kp / Ti x (NH4 - NH4_ref) x 5분   (포화 시 적분 중지)

sp_base 는 DO PI 와 같은 2.0 이다. 튜닝은 RL 과 같은 학습 날짜(건기 0 ~ 6일, 하루 reset)에서
"평균 NH4 가 DO PI 이하" 를 만족하는 조합 중 폭기 에너지가 가장 작은 것을 고른다.
"""
import numpy as np

from setpoint_env import SP_HIGH, SP_LOW, setpoint_to_action

SP_BASE = 2.0
STEP_H = 5 / 60


class ABAC:
    def __init__(self, nh4_ref, kp, ti_hours):
        self.nh4_ref, self.kp, self.ti = float(nh4_ref), float(kp), float(ti_hours)
        self.integral = 0.

    def reset(self):
        self.integral = 0.

    def __call__(self, obs, env=None):
        err = float(obs[0]) - self.nh4_ref  # obs[0] = 방류 NH4
        raw = SP_BASE + self.kp * err + self.integral
        sp = float(np.clip(raw, SP_LOW, SP_HIGH))
        if raw == sp or (raw > SP_HIGH and err < 0) or (raw < SP_LOW and err > 0):
            self.integral += self.kp / self.ti * err * STEP_H
        return [setpoint_to_action(sp)]

    def params(self):
        return {'nh4_ref': self.nh4_ref, 'kp': self.kp, 'ti_hours': self.ti}


GRID = [dict(nh4_ref=r, kp=k, ti_hours=t) for r in (1.0, 1.5, 2.0, 2.5, 3.0)
        for k in (0.25, 0.5, 1.0) for t in (1.0, 3.0)]
