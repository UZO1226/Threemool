"""e73 예산 속도 차단: 하루 NH4 누적 예산을 남은 스텝에 고르게 나눈 몫을 넘지 않게 공기를 올리기만 한다.

e68 비용은 그날 NH4 누적합이 380 (= 4 x 95) 을 넘은 뒤의 스텝마다 1 이다. 차단은 스텝 t 에서
- 몫 a_t = (380 - 누적합) / 남은 스텝 수 (누적합이 380 이하일 때). 이미 넘었으면 a_t = 4 (하루 평균 기준).
- 목표 = a_t - 여유. 여유는 정책을 보기 전에 공기 0 정책 + 차단으로 dev 에서 정한다 (protocol.json).
- 대리 모델 구성원 가운데 그 에피소드의 환경 구성원을 뺀 4개의 1스텝 NH4 예측 평균 (모델 오차 흉내, Park/e10 과 같은 방식)
  이 목표 이하가 되는 가장 작은 공기 격자값 (0 ~ 9,000, 250 간격) 을 필요 공기로 둔다. 후보는 환경과 같은 방식으로
  그 시각 기록 배분과 구간 상한을 적용한다. 만족하는 후보가 없으면 9,000 이다.
- 실행 공기 = max(정책 공기, 필요 공기). 올리지 않은 스텝은 정책 행동을 그대로 넘긴다 (차단 없는 평가와 비트 단위로 같게).

Park/code/min_brake.py 의 생각 (최소 개입, 다른 구성원 평균, 올리기만) 을 따르되 코드는 새로 썼다. 목표가 고정 3.5 가 아니라
남은 예산의 몫이라는 점이 다르다.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e68'))
import env68 as E  # noqa: E402
from darrow_env import AIR_MAX, action_to_air, air_to_action  # noqa: E402

GRID_STEP = 250.0


class PaceShield:
    def __init__(self, env, margin, members='others', grid_step=GRID_STEP):
        assert members in ('others', 'same')
        self.env, self.margin, self.members = env, float(margin), members
        self.grid = np.arange(0.0, AIR_MAX + 1e-9, grid_step)
        self.reset_counts()

    def reset_counts(self):
        self.steps = self.raised = self.infeasible = 0
        self.added_air = 0.0
        self.log = []   # 스텝별 (목표, 필요 공기 만족 여부, 선택 후보 예측 NH4) 는 검사에서만 쓴다

    def target(self):
        env = self.env
        left = env.T - env.t
        allow = (E.BUDGET - env.nh4_sum) / left
        return np.where(env.nh4_sum <= E.BUDGET, allow, E.NH4_LIMIT) - self.margin

    def predict_nh4(self):
        """후보 격자마다 다음 스텝 NH4 예측 (N, K)."""
        env, dyn = self.env, self.env.dyn
        t, N, K = env.t, len(env.member), len(self.grid)
        zone = np.minimum(self.grid[None, :, None] * env.x['frac'][:, t][:, None, :], env.cap[None, None, :]).reshape(N * K, -1)
        rep = lambda v: np.repeat(v, K, axis=0)  # noqa: E731
        state = rep(env.state)
        x = np.column_stack([state, rep(env.x['exo'][:, t]), zone / 1000.0, rep(env.prev_zone) / 1000.0])
        member = rep(env.member)
        total, weight = np.zeros(N * K), np.zeros(N * K)
        for m in range(dyn.n_members):
            rows = np.flatnonzero(member != m if self.members == 'others' else member == m)
            if len(rows):
                total[rows] += dyn.models[m][0].predict(x[rows])
                weight[rows] += 1.0
        return np.clip(state[:, 0] + total / weight, 0.0, dyn.hi[0]).reshape(N, K)

    def required(self):
        tgt = self.target()
        nh4 = self.predict_nh4()
        ok = nh4 <= tgt[:, None]
        has = ok.any(1)
        idx = ok.argmax(1)
        req = np.where(has, self.grid[idx], AIR_MAX)
        return req, has, tgt, nh4[np.arange(len(idx)), idx]

    def wrap(self, act):
        """act(obs) -> 행동 을 차단을 거친 행동으로 바꾼다."""
        def shielded(obs):
            a = np.asarray(act(obs), dtype=float).reshape(-1)
            req, has, tgt, pred = self.required()
            raise_ = req > action_to_air(a)
            self.steps += len(a)
            self.raised += int(raise_.sum())
            self.infeasible += int((~has).sum())
            self.added_air += float(np.where(raise_, req - action_to_air(a), 0.0).sum())
            if self.log is not None:
                self.log.append((tgt, has, pred, raise_))
            return np.where(raise_, air_to_action(req), a)
        return shielded

    def counts(self):
        return dict(intervention_pct=100.0 * self.raised / self.steps, infeasible_pct=100.0 * self.infeasible / self.steps,
                    added_air_per_step=self.added_air / self.steps)


def air0(obs):
    return np.full(len(obs), -1.0)
