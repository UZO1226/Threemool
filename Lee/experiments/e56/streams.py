"""Build separate within-update targets from immutable e55 training captures."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from targets import cost_rates, public_features


def read_stream(run, data, horizon, smoke=False):
    assert horizon in (1, 4)
    original = json.loads((run / 'correction/fit_summary.json').read_text())
    config = json.loads((run / 'effective_config.json').read_text())
    assert original['base_training_steps'] == 401280 and not original['smoke_only']
    assert not original['test_evaluated'] and original['checks']['passed']
    assert original['policy_seed'] == config['seed']
    assert hashlib.sha256(data.read_bytes()).hexdigest() == original['data_sha256']
    training_days = set(map(str, pd.read_pickle(data)['days']['train']))
    fit_updates = [1] if smoke else list(range(1, 53))
    held_updates = [2] if smoke else list(range(53, 67))
    subsets = {}
    hashes = {}
    for label, updates in (('fit', fit_updates), ('held', held_updates)):
        rows = []
        for update in updates:
            path = run / 'transitions' / f'update_{update:03d}.npz'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == original['capture_file_sha256'][path.name]
            hashes[path.name] = digest
            with np.load(path, allow_pickle=False) as saved:
                assert int(saved['update']) == update
                obs = saved['obs'].copy()
                action = saved['action'].copy()
                done = saved['done'].copy()
                days = saved['day'].copy()
                assert obs.shape == (760, 8, 11) and action.shape == done.shape == (760, 8)
                assert set(np.unique(days)).issubset(training_days)
                np.testing.assert_array_equal(done, saved['within_episode'] == 94)
                assert done[-1].all() and done.sum() == 64
                targets = []
                for physical, index in (('nh4', 0), ('do', 1)):
                    cost = saved[physical]
                    actual = saved['next_obs'][:, :, 0] > 4 if index == 0 else saved['next_obs'][:, :, 2:5].max(-1) > 2
                    np.testing.assert_array_equal(cost, actual)
                    rate, _ = cost_rates(cost, done, horizon)
                    targets.append(rate)
                rows.append({'x': public_features(obs.reshape(-1, 11), action.reshape(-1)),
                             'target': np.stack(targets, axis=-1).reshape(-1, 2),
                             'day': days.reshape(-1)})
        subsets[label] = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    assert set(fit_updates).isdisjoint(held_updates)
    return subsets['fit'], subsets['held'], {
        'fit_updates': fit_updates, 'held_updates': held_updates,
        'capture_file_sha256': hashes, 'policy_seed': config['seed'],
        'base_training_steps': 401280, 'extra_environment_training_steps': 0,
        'data_sha256': original['data_sha256'], 'horizon': horizon,
        'test_evaluated': False, 'smoke_only': smoke,
    }
