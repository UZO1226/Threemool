"""DARROW 최소 개입 차단: 예측 NH4 를 기준 아래로 유지하는 데 필요한 만큼만 공기량을 올린다 (Park 작성, Park/e10).

Park/e08, e09 의 차단은 위험할 때 공기량을 언제나 5,800 Nm3/h 로 올렸다. 이 차단은 대리 모델로 앞을 내다보고
"앞으로 horizon 스텝 동안 예측 NH4 최댓값 <= target" 을 만족하는 가장 작은 공기량을 찾아 정책 공기량과 비교한다.

- 후보 공기량은 0 ~ 9,000 을 grid_step 간격으로 나눈 격자다. 실행 공기량 = max(정책 공기량, 조건을 만족하는 가장
  작은 후보). 조건을 만족하는 후보가 없으면 9,000 이다
- 예측에는 관측한 현재 상태(관측 0 ~ 4번), 그 시각의 외부 입력과 구간 배분, 직전 구간 공기량을 쓴다. 미래 외부
  입력은 모르므로 horizon 동안 현재 값이 이어진다고 둔다 (지속 예측). 후보 공기량도 horizon 동안 같게 둔다
- members='others' 이면 환경이 그 에피소드에 쓰는 구성원을 빼고 나머지 4개 예측의 평균을 쓴다 (모델 오차를 흉내).
  members='same' 이면 환경과 같은 구성원을 쓴다 (정답 모델, 참고용)
- 대리 모델의 NH4 변화량은 공기량에 대해 비증가로 강제돼 있어 horizon 1 에서는 조건이 공기량에 대해 단조다.
  horizon 이 2 이상이면 DO 를 거치므로 단조가 보장되지 않는다
"""
import numpy as np

from darrow_env import AIR_MAX, STATE, action_to_air, air_to_action


class MinAirBrake:
    def __init__(self, dyn, target, horizon=1, members='others', grid_step=250.0):
        assert members in ('others', 'same')
        self.dyn, self.target, self.horizon, self.members = dyn, float(target), int(horizon), members
        self.grid = np.arange(0.0, AIR_MAX + 1e-9, grid_step)

    def _predict(self, x, member, n_states):
        """구성원 가중 평균 예측 변화량 (행, n_states)."""
        out = np.zeros((len(x), n_states))
        weight = np.zeros(len(x))
        for m in range(self.dyn.n_members):
            w = (member != m) if self.members == 'others' else (member == m)
            if not w.any():
                continue
            rows = np.flatnonzero(w)
            for j in range(n_states):
                out[rows, j] += self.dyn.models[m][j].predict(x[rows])
            weight[rows] += 1.0
        return out / weight[:, None]

    def required_air(self, obs, env):
        t, N, G = env.t, len(obs), len(self.grid)
        rep = lambda v: np.repeat(v, G, axis=0)  # noqa: E731  (에피소드 하나에 후보 G 개)
        state = rep(np.asarray(obs[:, :len(STATE)], dtype=float))
        exo, frac, zone_prev = rep(env.x['exo'][:, t]), rep(env.x['frac'][:, t]), rep(env.prev_zone)
        member = rep(env.member)
        zone = np.tile(self.grid, N)[:, None] * frac
        n_states = 1 if self.horizon == 1 else len(STATE)   # 한 스텝만 보면 NH4 모델만 필요하다
        worst = np.full(N * G, -np.inf)
        for _ in range(self.horizon):
            x = np.column_stack([state, exo, zone / 1000.0, zone_prev / 1000.0])
            delta = self._predict(x, member, n_states)
            nxt = state.copy()
            nxt[:, :n_states] = np.clip(state[:, :n_states] + delta, 0.0, self.dyn.hi[:n_states])
            state, zone_prev = nxt, zone
            worst = np.maximum(worst, state[:, 0])
        ok = (worst <= self.target).reshape(N, G)
        return np.where(ok.any(axis=1), self.grid[ok.argmax(axis=1)], AIR_MAX)

    def __call__(self, obs, a, env):
        policy_air = action_to_air(a)
        air = np.maximum(policy_air, self.required_air(obs, env))
        used = air > policy_air + 1e-9
        return np.where(used, air_to_action(air), a), used
