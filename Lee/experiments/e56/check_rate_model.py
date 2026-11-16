"""Synthetic inference and fixed-choice preflight, before controller outcomes."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from rate_model import ConstantRate, FastRate, RatePolicy, fit_models, slow_rates, ROOT
from targets import public_features


def main():
    rng = np.random.default_rng(5601)
    x = rng.normal(size=(4096, 12))
    x[:, 11] = rng.uniform(-1, 1, len(x))
    p = 1 / (1 + np.exp(-(.4 * x[:, 0] - .3 * x[:, 2:5].sum(-1) - .6 * x[:, 11])))
    y = np.column_stack((rng.binomial(4, p) / 4, rng.binomial(4, 1 - p) / 4))
    models = fit_models(x, y)
    fast = FastRate(models)
    probe = x[::-2].copy()
    raw = np.column_stack([model.predict(probe) for model in models])
    np.testing.assert_allclose(fast.raw(probe), raw, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fast.predict(probe), np.clip(raw, 0, 1), atol=1e-12, rtol=1e-12)
    restored = pickle.loads(pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(slow_rates(restored, probe), slow_rates(models, probe))
    worst = 0.
    for column, sign in {0: 1, 2: -1, 3: -1, 4: -1, 10: -1, 11: -1}.items():
        lower, upper = probe.copy(), probe.copy()
        lower[:, column], upper[:, column] = -1, 1
        difference = (fast.predict(upper) - fast.predict(lower)) * np.array([sign, -sign])
        worst = max(worst, float(np.maximum(-difference, 0).max()))
    assert worst <= 1e-12
    constants = fit_models(x, np.tile([0., .75], (len(x), 1)))
    assert all(isinstance(model, ConstantRate) for model in constants)
    np.testing.assert_array_equal(FastRate(constants).predict(probe), np.tile([0., .75], (len(probe), 1)))
    class Base:
        def act_det(self, obs):
            return np.clip(obs[:, 5:6], -1, 1)
    obs = x[:2048, :11].copy()
    base = Base()
    original = base.act_det(obs).reshape(-1)
    null = RatePolicy(base, constants)
    np.testing.assert_array_equal(null.act_det(obs), original)
    policy = RatePolicy(base, models)
    candidates = np.clip(original[:, None] + [0, -.125, .125, -.25, .25], -1, 1)
    independent = np.column_stack((np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
    np.testing.assert_array_equal(independent, public_features(np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
    scores = slow_rates(models, independent).sum(-1).reshape(-1, 5) + np.square(candidates - original[:, None])
    selected = candidates[np.arange(len(obs)), np.argmin(scores, axis=1)]
    np.testing.assert_array_equal(policy.act_det(obs), selected)
    target = json.loads(Path(__file__).with_name('target_check.json').read_text())
    assert target['passed']
    for name, expected in target['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
    result = {'passed': True, 'synthetic_only': True, 'test_evaluated': False,
              'raw_sklearn_max_difference': float(np.max(np.abs(fast.raw(probe) - raw))),
              'clipped_predictions_match_sklearn': True, 'serialized_predictions_exact': True,
              'coordinate_order_max_violation': worst, 'constant_target_fallback_exact': True,
              'flat_risk_null_action_identity': True, 'independent_fixed_candidate_choices_exact': True,
              'real_stream_target_preflight_passed': True,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                for name in ('rate_model.py', 'targets.py', 'check_rate_model.py')},
              'shared_inference_source_sha256': hashlib.sha256((ROOT / 'Lee/experiments/e55/fast_cost.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'scope': 'Synthetic model and action-choice checks plus previously verified real-stream targets. Not a DARROW fitted-regressor or controller performance result.'}
    Path(__file__).with_name('model_check.json').write_text(json.dumps(result, indent=2))
    print('E56_RATE_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
