"""초기 관측 범위, 회귀 정규화 및 추가 정보 gate를 검사한다."""
import numpy as np
import torch
from initial_signal import initial_features, fit_ridge, predict, assessment, signal_gate


class Toy(torch.nn.Module):
    def forward(self, state, exog, extra):
        return state + torch.tensor([.1, .2])


def main():
    state = torch.arange(288).float()[None, :, None].expand(2, -1, 2).clone()
    exog, extra = torch.ones(2, 288, 6), torch.zeros(2, 288, 8)
    x, errors = initial_features(Toy(), state, exog, extra)
    np.testing.assert_allclose(errors.mean(1), np.tile([.9, .8], (2, 1)), atol=1e-6)
    for z in (state, exog, extra): z[:, 13:] = 10000
    np.testing.assert_array_equal(x, initial_features(Toy(), state, exog, extra)[0])
    rng = np.random.default_rng(15000)
    x = rng.normal(size=(263, 8))
    y = 2*x[:, 6] - x[:, 7]
    result = assessment(x, y)
    rows = [dict(seed=seed, **result) for seed in (0, 1)]
    assert signal_gate(rows)['passed']
    no_increment = assessment(x, 2*x[:, 0])
    assert not signal_gate([dict(seed=seed, **no_increment) for seed in (0, 1)])['passed']
    fit = fit_ridge(x[:131], y[:131], list(range(8)))
    altered = x.copy(); altered[132:] *= 1e6
    assert fit == fit_ridge(altered[:131], y[:131], list(range(8)))
    assert np.isfinite(predict(fit, x)).all()
    constant = fit_ridge(x[:131], y[:131], [])
    np.testing.assert_allclose(predict(constant, x), y[:131].mean())
    print('PASS prefix-only observations, chronological folds, train-only scaling, constant control, incremental-signal gate')


if __name__ == '__main__': main()
