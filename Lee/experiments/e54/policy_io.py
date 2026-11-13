"""Load history policies with the same budget/source/freeze guards as e44."""
import json
from pathlib import Path
import sys

import torch

from history_agent import HistoryFCSRL
from history_env import evaluate_history
from train_history import METHODS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
from evaluate_pair import load_run as load_legacy, file_hashes, freeze_identities
from quantile_env import evaluate
from ppo_config import PPOConfig
from ppo_fcsrl import RepresentationConfig


def load_run(folder, freeze_path, allow_smoke=False):
    config = json.loads((folder / 'effective_config.json').read_text())
    if config['method'] not in METHODS:
        return load_legacy(folder, freeze_path, allow_smoke=allow_smoke)
    metrics = json.loads((folder / 'metrics.json').read_text())
    progress = json.loads((folder / 'progress.json').read_text())
    assert progress['phase'] == 'complete'
    metadata = config['experiment_metadata']
    assert metadata['test_evaluated'] is False and metadata['environment'] == 'new'
    assert metadata['upstream_commit'] == 'd882b91054d52c7a63a18a21f7db67bbbed82b63'
    assert metadata['terminal_reward_air_alias_corrected']
    assert metadata['policy_seed'] == config['seed']
    manifest = json.loads((folder.parent.parent / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        source = Path(name)
        if not source.exists():
            source = ROOT / name.replace('\\', '/').split('/Threemool/', 1)[1]
        assert expected in freeze_identities(source), f'Training source changed: {name}'
    assert metadata['environment_freeze_sha256'] in freeze_identities(freeze_path)
    c = PPOConfig.from_dict(config['config'])
    assert c.cost_limit == 12 and c.n_env == 8 and c.horizon == 760
    assert metrics['steps'] == c.actual_steps == (6080 if allow_smoke else 401280)
    representation = RepresentationConfig(**config['representation'])
    assert representation == RepresentationConfig()
    torch.manual_seed(0)
    agent = HistoryFCSRL(17, 1, config=c, repr_config=representation, mode=METHODS[config['method']])
    checkpoint = folder / 'model_final'
    hashes = file_hashes(checkpoint)
    agent.load(checkpoint)
    return agent, config, metrics, hashes


def score_policy(data, dynamics, agent, air_offset=0, split='val', draws=8):
    if isinstance(agent, HistoryFCSRL):
        return evaluate_history(data, dynamics, act_batch=lambda x: agent.act_det(x)[:, 0],
            air_offset=air_offset, split=split, draws=draws, mode=agent.history_mode)
    return evaluate(data, dynamics, act_batch=lambda x: agent.act_det(x)[:, 0],
        air_offset=air_offset, split=split, draws=draws)
