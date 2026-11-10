"""DARROW 최소 개입 차단: 예측 NH4 를 기준 아래로 유지하는 데 필요한 만큼만 공기량을 올린다 (Park 작성, Park/e10, e11).

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

do_cap=True (Park/e11) 이면 DO 상한도 함께 본다. 후보는 정책 공기량과 그보다 큰 격자 값이다 (공기량은 올리기만 한다).
1. 예측 NH4 최댓값 <= target 이고 예측 구간 DO 최댓값 <= 2 인 후보가 있으면 그중 가장 작은 공기량
2. 없으면 (NH4 를 지키려면 DO 를 넘는 충돌) 실제 비용 기준인 1[NH4 > 4] + 1[DO > 2] 의 예측값이 가장 작은 후보,
   같으면 가장 작은 공기량. target 은 여유분이라서, 충돌할 때는 실제 비용 기준을 따른다
do_cap=False 이면 Park/e10 과 같은 계산이다.

trim=True (Park/e14) 이면 공기량을 내리기도 한다 (양방향). 먼저 e10 처럼 NH4 에 필요한 최소 공기량 r 을 구한다.
1. 정책 공기량 < r 이면 r 로 올린다 (e10 과 같음, NH4 우선)
2. 정책 공기량 >= r 이고 정책 공기량에서 1스텝 뒤 예측 구간 DO 최댓값 > 2 이면, r 이상 정책 공기량 이하의 격자 값 중
   예측 DO <= 2 인 가장 큰 공기량으로 내린다. 없으면 r 로 내린다. r 이상이므로 예측 NH4 <= target 은 그대로 지킨다
예측 DO 는 정책 공기량에서 먼저 계산하고, 넘는 에피소드만 격자 전체로 다시 계산한다. lowered_steps, steps 에 내린 횟수를 센다.
"""
import numpy as np

from darrow_env import AIR_MAX, DO_MAX, NH4_MAX, STATE, action_to_air, air_to_action


class MinAirBrake:
    def __init__(self, dyn, target, horizon=1, members='others', grid_step=250.0, do_cap=False, trim=False):
        assert members in ('others', 'same')
        self.dyn, self.target, self.horizon, self.members = dyn, float(target), int(horizon), members
        self.grid = np.arange(0.0, AIR_MAX + 1e-9, grid_step)
        self.do_cap, self.trim = bool(do_cap), bool(trim)
        assert not (self.do_cap and self.trim), 'do_cap 과 trim 은 하나만 고른다'
        assert not self.trim or self.horizon == 1, 'trim 은 1스텝만 쓴다'
        self.lowered_steps, self.steps = 0, 0

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

    def _rollout(self, obs, env, cand, n_states):
        """후보 공기량 cand (N, K) 를 horizon 동안 유지했을 때 예측 NH4 최댓값과 구간 DO 최댓값 (N, K)."""
        t, (N, K) = env.t, cand.shape
        rep = lambda v: np.repeat(v, K, axis=0)  # noqa: E731  (에피소드 하나에 후보 K 개)
        state = rep(np.asarray(obs[:, :len(STATE)], dtype=float))
        exo, frac, zone_prev = rep(env.x['exo'][:, t]), rep(env.x['frac'][:, t]), rep(env.prev_zone)
        member = rep(env.member)
        zone = cand.reshape(-1)[:, None] * frac
        worst_nh4 = np.full(N * K, -np.inf)
        worst_do = np.full(N * K, -np.inf)
        for _ in range(self.horizon):
            x = np.column_stack([state, exo, zone / 1000.0, zone_prev / 1000.0])
            delta = self._predict(x, member, n_states)
            nxt = state.copy()
            nxt[:, :n_states] = np.clip(state[:, :n_states] + delta, 0.0, self.dyn.hi[:n_states])
            state, zone_prev = nxt, zone
            worst_nh4 = np.maximum(worst_nh4, state[:, 0])
            if n_states == len(STATE):
                worst_do = np.maximum(worst_do, state[:, 2:].max(axis=1))
        return worst_nh4.reshape(N, K), worst_do.reshape(N, K)

    def required_air(self, obs, env):
        N = len(obs)
        n_states = 1 if self.horizon == 1 else len(STATE)   # 한 스텝만 보면 NH4 모델만 필요하다
        nh4, _ = self._rollout(obs, env, np.tile(self.grid, (N, 1)), n_states)
        ok = nh4 <= self.target
        return np.where(ok.any(axis=1), self.grid[ok.argmax(axis=1)], AIR_MAX)

    def do_aware_air(self, obs, env, policy_air):
        grid = self.grid[None, :]
        cand = np.column_stack([policy_air, np.where(grid > policy_air[:, None], grid, policy_air[:, None])])
        nh4, do = self._rollout(obs, env, cand, len(STATE))
        both = (nh4 <= self.target) & (do <= DO_MAX)
        step_cost = (nh4 > NH4_MAX).astype(int) + (do > DO_MAX).astype(int)
        pick = np.where(both.any(axis=1), both.argmax(axis=1), step_cost.argmin(axis=1))
        return cand[np.arange(len(cand)), pick]

    def trimmed_air(self, obs, env, policy_air):
        r = self.required_air(obs, env)
        air = np.maximum(policy_air, r)
        rows = np.flatnonzero(policy_air >= r)
        if len(rows):
            sub = _Rows(env, rows)
            _, do = self._rollout(obs[rows], sub, policy_air[rows, None], len(STATE))
            over = rows[do[:, 0] > DO_MAX]
            if len(over):
                _, do_grid = self._rollout(obs[over], _Rows(env, over), np.tile(self.grid, (len(over), 1)), len(STATE))
                inside = (self.grid[None, :] >= r[over, None]) & (self.grid[None, :] <= policy_air[over, None])
                ok = inside & (do_grid <= DO_MAX)
                best = np.where(ok, self.grid[None, :], -np.inf).max(axis=1)
                air[over] = np.where(np.isfinite(best), best, r[over])
        return air

    def __call__(self, obs, a, env):
        policy_air = action_to_air(a)
        if self.do_cap:
            air = self.do_aware_air(obs, env, policy_air)
        elif self.trim:
            air = self.trimmed_air(obs, env, policy_air)
        else:
            air = np.maximum(policy_air, self.required_air(obs, env))
        used = np.abs(air - policy_air) > 1e-9
        self.lowered_steps += int((air < policy_air - 1e-9).sum())
        self.steps += len(air)
        return np.where(used, air_to_action(air), a), used


class _Rows:
    """환경의 일부 에피소드만 보는 얇은 보기 (_rollout 이 쓰는 t, x, prev_zone, member 만)."""

    def __init__(self, env, rows):
        self.t, self.member, self.prev_zone = env.t, env.member[rows], env.prev_zone[rows]
        self.x = {k: env.x[k][rows] for k in ('exo', 'frac')}
