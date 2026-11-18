"""Lee/e08 MLP 대리 모델 (darrow_surrogate_mlp.npz) 을 GBM Dynamics 와 같은 모양으로 부르는 연결 코드 (Park 작성, Park/e38).

MinAirBrake 는 dyn.models[m][j].predict(x) (구성원 m, 상태 j) 와 dyn.hi, dyn.n_members, dyn.step 을 쓴다.
MLP 는 한 번에 상태 5개를 예측하므로, 같은 x 로 j 를 바꿔 부르면 마지막 계산을 다시 쓴다.
입력 x 는 GBM 과 같다: [상태 5, 외생 9, 구역 공기량/1000 5, 직전 구역 공기량/1000 5] (24 열).
"""
import numpy as np


def _silu(h):
    return h * (1 / (1 + np.exp(-np.clip(h, -30, 30))))


class _Head:
    def __init__(self, net, m, j):
        self.net, self.m, self.j = net, m, j

    def predict(self, x):
        return self.net.delta(x, self.m)[:, self.j]


class MLPSurrogate:
    def __init__(self, path):
        z = np.load(path)
        self.xm, self.xs, self.ym, self.ys = z['xm'], z['xs'], z['ym'], z['ys']
        self.hi = np.asarray(z['hi'])
        self.W = [z[f'W{i}'].astype(float) for i in range(3)]
        self.b = [z[f'b{i}'].astype(float) for i in range(3)]
        self.n_members = self.W[0].shape[0]
        self.models = [[_Head(self, m, j) for j in range(len(self.ym))] for m in range(self.n_members)]
        self._last = (None, None, None)   # (x 객체, 구성원, 변화량)

    def delta(self, x, m):
        """x (N, 24), 구성원 번호 m -> 변화량 (N, 5)."""
        lx, lm, ld = self._last
        if lx is x and lm == m:
            return ld
        h = (np.asarray(x, dtype=float) - self.xm) / self.xs
        for i in range(3):
            h = h @ self.W[i][m] + self.b[i][m]
            if i < 2:
                h = _silu(h)
        d = h * self.ys + self.ym
        self._last = (x, m, d)
        return d

    def step(self, state, exo, zone, zone_prev, member):
        x = np.column_stack([state, exo, zone / 1000.0, zone_prev / 1000.0])
        delta = np.zeros_like(state, dtype=float)
        for m in np.unique(member):
            rows = member == m
            delta[rows] = self.delta(x[rows], int(m))
        return np.clip(state + delta, 0.0, self.hi)


class _MixHead:
    def __init__(self, g, n, w):
        self.g, self.n, self.w = g, n, w

    def predict(self, x):
        return (1.0 - self.w) * self.g.predict(x) + self.w * self.n.predict(x)


class MixedSurrogate:
    """차단 예측용: 구성원 m, 상태 j 마다 (1 - w) x GBM + w x MLP (Park/e38 모델 불일치 단계).
    상태 상한은 환경과 같은 GBM 의 hi 를 쓴다. 'others' 로 빼는 구성원 번호는 GBM, MLP 에 같은 번호를 쓴다."""

    def __init__(self, gbm, mlp, w):
        assert gbm.n_members == mlp.n_members
        self.w, self.hi, self.n_members = float(w), np.asarray(gbm.hi), gbm.n_members
        self.models = [[_MixHead(gbm.models[m][j], mlp.models[m][j], self.w) for j in range(len(self.hi))]
                       for m in range(self.n_members)]


class _MaskHead:
    def __init__(self, h, cols, vals):
        self.h, self.cols, self.vals = h, cols, vals

    def predict(self, x):
        x = np.array(x, dtype=float)
        x[:, self.cols] = self.vals
        return self.h.predict(x)


class MaskedSurrogate:
    """차단 예측용: 입력 열 cols 를 고정값 vals 로 바꿔 예측한다 (Park/e39 정보 제한, 측정할 수 없는 유입 변수를 학습 구간 평균으로)."""

    def __init__(self, dyn, cols, vals):
        self.hi, self.n_members = np.asarray(dyn.hi), dyn.n_members
        cols, vals = np.asarray(cols, dtype=int), np.asarray(vals, dtype=float)
        self.models = [[_MaskHead(dyn.models[m][j], cols, vals) for j in range(len(self.hi))] for m in range(self.n_members)]
