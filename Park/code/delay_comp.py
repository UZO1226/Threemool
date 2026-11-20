"""NH4 분석계 지연 보정 (Park 작성, Park/e53). Smith 예측기 방식의 지연 보정이다.

분석계가 d 스텝 늦은 NH4 를 줄 때, 늦은 값에서 출발해 그 뒤 d 스텝 동안 실제로 넣은 구간 공기량, 지연 없는 다른 센서 값
(NO3, DO 3개), 그 시각의 외부 입력으로 대리 모델 (환경이 쓰는 구성원을 뺀 4개 평균, 대표 차단과 같음) 을 굴려 지금 NH4 를 추정한다.
정책과 차단은 추정한 NH4 를 본다. 쓰는 정보는 대표 차단과 같다 (자기가 넣은 공기량, 외부 입력, 다른 센서).

    comp = DelayComp(env, dyn, d)
    o2 = comp(o)   # 매 스텝 한 번, o 의 0번이 d 스텝 늦은 NH4
"""
import numpy as np

from darrow_env import NH4_MAX, STATE
from min_brake import MinAirBrake


class DelayComp:
    def __init__(self, env, dyn, delay):
        self.env, self.dyn, self.d = env, dyn, int(delay)
        self._predict = MinAirBrake(dyn, NH4_MAX, 1, 'others')._predict
        self.obs_hist, self.zones = [], []

    def __call__(self, o):
        env, t = self.env, self.env.t
        if t == 0:
            self.obs_hist, self.zones = [], [env.prev_zone.copy()]   # zones[k + 1] = 스텝 k 의 구간 공기량, zones[0] = 시작 전
        else:
            self.zones.append(env.prev_zone.copy())
        o = np.asarray(o, dtype=float).copy()
        self.obs_hist.append(o.copy())
        if self.d == 0:
            return o
        n = o[:, 0].copy()                    # = 실제 NH4 (max(t - d, 0) 시각)
        for k in range(max(t - self.d, 0), t):
            state = self.obs_hist[k][:, :len(STATE)].copy()
            state[:, 0] = n
            x = np.column_stack([state, env.x['exo'][:, k], self.zones[k + 1] / 1000.0, self.zones[k] / 1000.0])
            n = np.clip(n + self._predict(x, env.member, 1)[:, 0], 0.0, self.dyn.hi[0])
        o[:, 0] = n
        return o
