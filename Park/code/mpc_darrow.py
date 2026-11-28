"""DARROW 짧은 예측 구간 모델 예측 제어 (Park 작성, Park/e48). 학습 정책이 없는 비학습 기준이다.

대표 차단과 같은 정보만 쓴다: 관측한 지금 상태 (관측 0 ~ 4번), 그 시각의 외부 입력과 구간 배분 (예측 구간 동안 그대로 둠,
지속 예측), 직전 구간 공기량, 환경이 쓰는 구성원을 뺀 4개 구성원의 평균 예측 (`MinAirBrake._predict` 를 그대로 씀).

매 스텝 후보 공기량 열 (첫 스텝 u0, 나머지 H-1 스텝 u1, 이동 묶음) 을 H 스텝 굴려 다음 목적이 가장 작은 후보의 u0 를 실행한다.

    J = sum_h [ 1(NH4_h + margin > 4) + 1(구간 DO 최댓값_h > do_limit) + lam x 공기량_h / 9000 ]

위반 두 항은 환경 비용 (NH4 위반 + DO 위반 스텝 수) 과 같은 모양이다. 같은 값이면 공기량이 작은 후보를 고른다 (후보를 u0, u1 오름차순으로
두고 첫 최솟값을 고름).

objective='excess' (Park/e52) 는 위반 지표 대신 초과량을 쓴다. 피할 수 없는 위반 앞에서도 공기를 더 넣으면 목적이 줄어든다.

    J = sum_h [ max(NH4_h + margin - 4, 0) + max(구간 DO 최댓값_h - do_limit, 0) + lam x 공기량_h / 9000 ]
"""
import numpy as np

from darrow_env import AIR_MAX, NH4_MAX, STATE, air_to_action
from min_brake import MinAirBrake


class ShortMPC:
    def __init__(self, env, dyn, H=4, margin=0.2, lam=0.5, do_limit=1.8, step0=500.0, step1=1500.0, objective='indicator'):
        self.env, self.dyn, self.H = env, dyn, int(H)
        self.margin, self.lam, self.do_limit = float(margin), float(lam), float(do_limit)
        assert objective in ('indicator', 'excess')
        self.objective = objective
        g0 = np.arange(0.0, AIR_MAX + 1e-9, step0)
        g1 = np.arange(0.0, AIR_MAX + 1e-9, step1)
        self.cand = np.array([(a, b) for a in g0 for b in g1])          # (K, 2), u0 다음 u1 오름차순
        self._predict = MinAirBrake(dyn, NH4_MAX, 1, 'others')._predict

    def plan(self, obs):
        env, (K, N) = self.env, (len(self.cand), len(obs))
        t = env.t
        rep = lambda v: np.repeat(v, K, axis=0)  # noqa: E731
        state = rep(np.asarray(obs[:, :len(STATE)], dtype=float))
        exo, frac, zone_prev, member = rep(env.x['exo'][:, t]), rep(env.x['frac'][:, t]), rep(env.prev_zone), rep(env.member)
        u0, u1 = np.tile(self.cand[:, 0], N), np.tile(self.cand[:, 1], N)
        J = np.zeros(N * K)
        for h in range(self.H):
            air = u0 if h == 0 else u1
            zone = air[:, None] * frac
            x = np.column_stack([state, exo, zone / 1000.0, zone_prev / 1000.0])
            state = np.clip(state + self._predict(x, member, len(STATE)), 0.0, self.dyn.hi)
            zone_prev = zone
            nx, dx = state[:, 0] + self.margin - NH4_MAX, state[:, 2:].max(axis=1) - self.do_limit
            if self.objective == 'indicator':
                J += (nx > 0) + (dx > 0) + self.lam * air / AIR_MAX
            else:
                J += np.maximum(nx, 0.0) + np.maximum(dx, 0.0) + self.lam * air / AIR_MAX
        return self.cand[J.reshape(N, K).argmin(axis=1), 0]

    def __call__(self, obs):
        return air_to_action(self.plan(obs))
