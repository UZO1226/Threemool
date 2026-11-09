"""기준 제어 (Park 작성).

- 고정 KLa5: 정책이 항상 같은 행동을 낸다
- DO PI: 5번 조 DO 를 2 mg/L 로 맞추는 PI 제어. BSM1 원본 reginit.m 의 계수
  (K = 25, Ti = 0.002 day, Tt = 0.001 day, 출력 0 ~ 240, 기준 출력 144) 를 쓰고 1분마다 계산한다.
  정확한 DO 센서를 가정한다.
"""
import numpy as np

from env import KLA_MAX, kla_to_action


def fixed_kla(kla):
    """항상 KLa5 = kla 를 내는 정책."""
    action = np.array([kla_to_action(kla)])
    return lambda obs: action


class DOPIController:
    SETPOINT = 2.0
    GAIN = 25.0
    TI = 0.002        # day
    TT = 0.001        # day, anti windup
    OFFSET = 144.0
    DT = 1 / 1440     # 1분

    def __init__(self):
        self.integral = 0.0

    def __call__(self, do):
        error = self.SETPOINT - do
        raw = self.OFFSET + self.GAIN * error + self.integral
        out = float(np.clip(raw, 0.0, KLA_MAX))
        self.integral += (self.GAIN / self.TI * error + (out - raw) / self.TT) * self.DT
        return out


# 이름, 정책, controller. controller 가 있으면 정책 행동은 무시된다.
BASELINES = {
    'fixed84': (fixed_kla(84.0), None),
    'fixed240': (fixed_kla(240.0), None),
    'do_pi': (fixed_kla(120.0), DOPIController),
}
