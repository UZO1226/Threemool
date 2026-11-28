"""Add a monotone next-DO upper guard to the fixed matched-reference geometry."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e74'))
from anchor_policy import AnchorPolicy, configuration as anchor_configuration, project, public_features, eligible, calibrate
from rate_model import FastRate
from upper_forecast import upper_prediction


def configuration(low, high, weight, reference_offset, forecast_seed, forecast_record, enabled=True, shield_enabled=True):
    config = anchor_configuration(low, high, weight, reference_offset, enabled)
    assert type(shield_enabled) is bool and type(forecast_seed) is int and 0 <= forecast_seed <= 4
    assert forecast_record['policy_seed'] == forecast_seed and forecast_record['admission']['passed']
    config['do_shield'] = {'enabled': shield_enabled, 'policy_seed': forecast_seed,
        'model_sha256': forecast_record['model_sha256'], 'upper_config': forecast_record['config']}
    return config


class ShieldPolicy(AnchorPolicy):
    def __init__(self, base, models, config, forecast_model=None):
        plain = {key: value for key, value in config.items() if key != 'do_shield'}
        super().__init__(base, models, plain)
        shield = config['do_shield']
        if set(shield) != {'enabled', 'policy_seed', 'model_sha256', 'upper_config'} or type(shield['enabled']) is not bool:
            raise ValueError('Invalid forecast shield specification')
        seed = shield['policy_seed']
        if type(seed) is not int or not 0 <= seed <= 4:
            raise ValueError('Invalid forecast policy seed')
        if forecast_model is None:
            path = ROOT.parent / f'results/e76_do_forecast/seed_{seed}/do_model.pkl'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == shield['model_sha256']
            with path.open('rb') as handle:
                forecast_model = pickle.load(handle)
        self.forecast = FastRate([forecast_model])
        self.original_forecast = forecast_model
        self.config = json.loads(json.dumps(config))

    def upper(self, obs, action, original=False):
        x = public_features(obs, action)
        current = np.asarray(obs)[:, 2:5].max(-1)
        change = self.original_forecast.predict(x) if original else self.forecast.raw(x)[:, 0]
        return upper_prediction(np.maximum(current + change, 0.), current, self.config['do_shield']['upper_config'])

    def geometry(self, obs):
        candidates, risk = super().geometry(obs)
        if not self.config['do_shield']['enabled']:
            return candidates, risk
        upper = self.upper(np.repeat(obs, 9, axis=0), candidates.reshape(-1)).reshape(-1, 9)
        valid = upper <= np.maximum(2., upper[:, :1]) + 1e-12
        return np.where(valid, candidates, candidates[:, :1]), np.where(valid, risk, risk[:, :1])


def load_policy(base, models, path):
    return ShieldPolicy(base, models, json.loads(Path(path).read_text()))
