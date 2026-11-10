"""혼합 식, 경계, 변화량 어댑터와 공기 단조성의 작은 검산."""
import numpy as np
from regime import NextNH4


class Predictor:
    def __init__(self, bias, slope=0):
        self.bias, self.slope = bias, slope

    def predict(self, x):
        return self.bias - self.slope * x[:, 14]


class Gate:
    def predict_proba(self, x):
        p = 1 / (1 + np.exp(x[:, 14]))
        return np.column_stack((1-p, p))


def main():
    x = np.zeros((101, 44)); x[:, 0] = 2; x[:, 14] = np.linspace(-3, 3, 101)
    model = NextNH4(15, gate=Gate(), normal=Predictor(2, .2), violation=Predictor(8, .3))
    nxt = model.predict_next(x)
    p, low, high = model.last_parts.T
    assert np.allclose(nxt, low+p*(high-low))
    assert np.allclose(model.predict(x)+x[:, 0], nxt)
    assert (np.diff(nxt) <= 0).all()
    # Even out-of-range expert values preserve ordering after separate clipping.
    model = NextNH4(15, gate=Gate(), normal=Predictor(20), violation=Predictor(-5))
    assert np.array_equal(model.predict_next(x), np.full(101, 4.))
    model = NextNH4(15, single=Predictor(50))
    assert np.array_equal(model.predict_next(x), np.full(101, 15.))
    print('PASS: mixture identity, ordered supports, delta adapter and monotonicity')


if __name__ == '__main__':
    main()
