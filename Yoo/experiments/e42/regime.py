"""실제 정답 대신 입력에서 추정한 위험도로 두 NH4 예측을 합친다."""
import numpy as np


class NextNH4:
    def __init__(self, high, single=None, gate=None, normal=None, violation=None):
        self.high = float(high)
        self.single = single
        self.gate = gate
        self.normal = normal
        self.violation = violation
        self.last_parts = None
        assert self.high > 4

    def predict_next(self, x):
        if self.single is not None:
            return np.clip(self.single.predict(x), 0, self.high)
        p = self.gate.predict_proba(x)[:, 1]
        low = np.clip(self.normal.predict(x), 0, 4)
        high = np.clip(self.violation.predict(x), 4, self.high)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        self.last_parts = np.column_stack((p, low, high))
        return (1 - p) * low + p * high

    def predict(self, x):
        # HistoryDynamics adds current NH4 to this return value.
        return self.predict_next(x) - x[:, 0]
