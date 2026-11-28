"""공기량에 따라 달라지는 온라인 NH4 오차 보정 차단 (Park 작성, Park/e42).

MinAirBrake 의 bias (Park/e31, 상수 보정 b) 대신, 지난 스텝의 1 스텝 NH4 예측 오차 e (관측 빼기 예측, 실제로 쓴 공기량 u)
를 e = b0 + b1 z, z = (u - REF) / 1000 으로 보고 에피소드마다 망각 계수 lam 의 재귀 최소제곱 (RLS) 으로 (b0, b1) 을 갱신한다.
1 스텝 여유 검사에서 격자 후보 공기량 g 의 예측에 b0 + b1 (g - REF) / 1000 을 더한다. 회복 검사, 회복 계획, DO 예측은
바꾸지 않는다 (bias 와 같음). b1 은 [b1_lo, b1_hi] 로 자른다 (기본 [-0.1, 0]: 보정은 모델의 공기 효과를 키우는 쪽만 허용).
P 의 대각합이 2 p0 를 넘으면 2 p0 로 줄인다 (자극이 적을 때 P 가 커지는 것을 막는다). 에피소드 첫 스텝에 되돌린다.
"""
import numpy as np

from darrow_env import AIR_MAX, NH4_MAX, STATE, action_to_air
from min_brake import MinAirBrake

REF = 5000.0


def rls_step(theta, P, phi, e, lam, b1_lo, b1_hi, p0):
    """에피소드별 RLS 한 스텝. theta (N, 2), P (N, 2, 2), phi (N, 2), e (N,). 새 (theta, P)."""
    Pphi = np.einsum('nij,nj->ni', P, phi)
    K = Pphi / (lam + np.einsum('ni,ni->n', phi, Pphi))[:, None]
    theta = theta + K * (e - np.einsum('ni,ni->n', phi, theta))[:, None]
    P = (P - np.einsum('ni,nj->nij', K, Pphi)) / lam
    tr = P[:, 0, 0] + P[:, 1, 1]
    P = P * np.minimum(1.0, 2.0 * p0 / tr)[:, None, None]
    theta[:, 1] = np.clip(theta[:, 1], b1_lo, b1_hi)
    return theta, P


class SlopeBiasBrake(MinAirBrake):
    def __init__(self, dyn, *args, slope=(0.95, -0.1, 0.0, 1.0), **kw):
        kw['bias'] = slope[0]   # MinAirBrake 의 갱신 시점과 last_mu 계산을 그대로 쓴다
        super().__init__(dyn, *args, **kw)
        self.lam, self.b1_lo, self.b1_hi, self.p0 = map(float, slope)
        self.theta, self.P, self.last_air = None, None, None

    def _update_bias(self, obs, env):
        N = len(obs)
        if env.t == 0 or self.theta is None or len(self.theta) != N:
            self.theta, self.P = np.zeros((N, 2)), np.tile(np.eye(2) * self.p0, (N, 1, 1))
            self.last_mu, self.last_air = None, None
        if self.last_mu is not None:
            e = np.asarray(obs[:, 0], dtype=float) - self.last_mu
            phi = np.column_stack([np.ones(N), (self.last_air - REF) / 1000.0])
            self.theta, self.P = rls_step(self.theta, self.P, phi, e, self.lam, self.b1_lo, self.b1_hi, self.p0)

    def _adaptive_ok(self, obs, env, rows=None):
        t, N, K = env.t, len(obs), len(self.grid)
        rep = lambda v: np.repeat(v, K, axis=0)  # noqa: E731
        state = rep(np.asarray(obs[:, :len(STATE)], dtype=float))
        zone = np.tile(self.grid, N)[:, None] * rep(env.x['frac'][:, t])
        x = np.column_stack([state, rep(env.x['exo'][:, t]), zone / 1000.0, rep(env.prev_zone) / 1000.0])
        pred = np.stack([np.clip(state[:, 0] + self.dyn.models[m][0].predict(x), 0.0, self.dyn.hi[0])
                         for m in range(self.dyn.n_members)])
        keep = np.arange(self.dyn.n_members)[:, None] != rep(env.member)[None, :]
        n = keep.sum(axis=0)
        mu = np.where(keep, pred, 0.0).sum(axis=0) / n
        sigma = np.sqrt(np.where(keep, (pred - mu) ** 2, 0.0).sum(axis=0) / n)
        if self.theta is not None:
            th = self.theta if rows is None else self.theta[rows]
            mu = mu + rep(th[:, 0]) + rep(th[:, 1]) * (np.tile(self.grid, N) - REF) / 1000.0
        margin = np.full(len(mu), self.margins[-1][1])
        for upper, m in reversed(self.margins):
            margin = np.where(sigma < upper, m, margin)
        if self.extra is not None:
            margin = margin + rep(self.extra if rows is None else self.extra[rows])
        return (mu + margin <= NH4_MAX).reshape(N, K)

    def __call__(self, obs, a, env):
        out, used = super().__call__(obs, a, env)
        self.last_air = np.clip(action_to_air(out), 0.0, AIR_MAX)
        return out, used
