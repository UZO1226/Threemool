"""Read authenticated next-state DO labels without adding future inputs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e56'))
from targets import public_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_blocks(run, data):
    config = json.loads((run / 'effective_config.json').read_text())
    original = json.loads((run / 'correction/fit_summary.json').read_text())
    assert original['checks']['passed'] and original['base_training_steps'] == 401280
    assert not original['test_evaluated'] and not original['smoke_only']
    assert original['policy_seed'] == config['seed'] and sha(data) == original['data_sha256']
    training_days = set(map(str, pd.read_pickle(data)['days']['train']))
    assert len(training_days) == 188
    groups = {'fit': list(range(1, 41)), 'calibration': list(range(41, 53)), 'held': list(range(53, 67))}
    blocks, hashes = {}, {}
    for label, updates in groups.items():
        rows = []
        for update in updates:
            path = run / 'transitions' / f'update_{update:03d}.npz'
            digest = sha(path)
            assert digest == original['capture_file_sha256'][path.name]
            hashes[path.name] = digest
            with np.load(path, allow_pickle=False) as saved:
                assert int(saved['update']) == update
                obs, future, action = saved['obs'], saved['next_obs'], saved['action']
                assert obs.shape == future.shape == (760, 8, 11)
                assert action.shape == saved['done'].shape == saved['day'].shape == (760, 8)
                assert np.isfinite(future).all()
                np.testing.assert_array_equal(saved['done'], saved['within_episode'] == 94)
                assert saved['done'][-1].all() and saved['done'].sum() == 64
                assert set(np.unique(saved['day'])).issubset(training_days)
                target = future[:, :, 2:5].max(-1)
                np.testing.assert_array_equal(saved['do'], target > 2)
                np.testing.assert_array_equal(saved['nh4'], future[:, :, 0] > 4)
                x = public_features(obs.reshape(-1, 11), action.reshape(-1))
                np.testing.assert_array_equal(x[:, :11], obs.reshape(-1, 11))
                assert np.isfinite(target).all() and np.all(target >= 0)
                rows.append({'x': x, 'y': target.reshape(-1).copy(), 'day': saved['day'].reshape(-1).copy()})
        blocks[label] = {key: np.concatenate([row[key] for row in rows]) for key in ('x', 'y', 'day')}
        assert len(blocks[label]['y']) == len(updates) * 6080
    assert [len(blocks[name]['y']) for name in groups] == [243200, 72960, 85120]
    return blocks, {'policy_seed': config['seed'], 'data_sha256': sha(data),
        'capture_file_sha256': hashes, 'update_partition': groups,
        'input_dimension': 12, 'input_rule': 'Current obs11 plus logged executed action, no future input.',
        'target_rule': 'Maximum of true next_obs DO1/DO2/DO3, including terminal observations.',
        'test_evaluated': False, 'extra_environment_training_steps': 0}
