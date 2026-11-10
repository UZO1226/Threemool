"""증강 자료가 미래 실측을 입력에 넣지 않는지 작은 합성 경로로 확인한다."""
import numpy as np
from experiment import rollout_examples


class ToyDynamics:
    n_members = 2

    def step(self, state, exo, zone, previous_zone, members, history):
        return state + (members + 1)[:, None] * .1


def main():
    state = np.arange(96)[:, None] + np.arange(5)[None, :]
    days = {'d': {'state': state.astype(float), 'exo': np.zeros((96, 9)), 'zone': np.ones((96, 5)) * 1000}}
    x, y = rollout_examples(['d'], days, ToyDynamics())
    assert x.shape == (95, 2, 44) and y.shape == (95, 2)
    assert np.allclose(x[1, :, 0], [.1, .2])
    assert np.allclose(y[1], [1.9, 1.8])
    assert np.allclose(x[2, :, 24], [.1, .2])
    assert np.allclose(x[2, :, 29], [0, 0])
    assert np.allclose(x[12, :, 0], [12, 12])
    assert np.allclose(x[:, :, 0] + y, state[1:, 0, None])
    changed = {'d': {key: value.copy() for key, value in days['d'].items()}}
    changed['d']['state'][1:12] = 999
    xx, yy = rollout_examples(['d'], changed, ToyDynamics())
    assert np.array_equal(x[:12], xx[:12])
    assert not np.array_equal(y[:12], yy[:12])
    assert np.array_equal(x[16:], xx[16:])
    print('PASS: recursive input order, target residual, 12-step reset, future-label isolation')


if __name__ == '__main__':
    main()
