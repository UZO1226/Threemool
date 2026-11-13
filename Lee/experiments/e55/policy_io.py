"""Load captured base policies and fixed correction models with provenance guards."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/code/darrow'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
from evaluate_pair import load_run as load_reference, file_hashes, freeze_identities
from quantile_env import evaluate, frozen_digest
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig
from ppo_config import PPOConfig
from fast_cost import FastCorrectedPolicy


def verify_sources(mapping, root):
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        assert expected in freeze_identities(path), f'Source changed: {path}'


def load_captured(run, freeze, reference, smoke=False):
    config = json.loads((run / 'effective_config.json').read_text())
    metrics = json.loads((run / 'metrics.json').read_text())
    progress = json.loads((run / 'progress.json').read_text())
    assert progress['phase'] == 'complete'
    meta = config['experiment_metadata']
    assert meta['environment'] == 'new' and not meta['test_evaluated']
    assert meta['terminal_reward_air_alias_corrected']
    assert meta['environment_freeze_sha256'] in freeze_identities(freeze)
    verify_sources(meta['source_hashes'], ROOT)
    capture = meta['passive_capture']
    assert capture['extra_environment_steps'] == 0
    verify_sources(capture['sources'], Path(__file__).parent)
    c = PPOConfig.from_dict(config['config'])
    assert c.n_env == 8 and c.horizon == 760 and c.cost_limit == 12
    assert progress['steps'] == metrics['steps'] == c.actual_steps == (6080 if smoke else 401280)
    if not smoke:
        manifest = json.loads((run.parent.parent / 'manifest.json').read_text())
        verify_sources(manifest['source_hashes'], ROOT)
    reference_config = json.loads((reference / 'effective_config.json').read_text())
    reference_metrics = json.loads((reference / 'metrics.json').read_text())
    assert config['method'] == reference_config['method'] == 'ppolag_fcsrl'
    assert config['seed'] == reference_config['seed']
    assert config['config'] == reference_config['config']
    assert config['representation'] == reference_config['representation']
    hashes = file_hashes(run / 'model_final')
    assert hashes == file_hashes(reference / 'model_final'), 'Passive collector changed original final checkpoint'
    assert metrics['val_final'] == reference_metrics['val_final']
    torch.manual_seed(0)
    agent = PPOFCSRLAgent(11, 1, config=c, repr_config=RepresentationConfig(**config['representation']))
    agent.load(run / 'model_final')
    return agent, config, metrics, hashes


def load_correction(folder, config, smoke=False):
    summary = json.loads((folder / 'fit_summary.json').read_text())
    assert summary['policy_seed'] == config['seed']
    assert not summary['test_evaluated'] and summary['smoke_only'] == smoke
    assert summary['extra_environment_steps'] == 0
    assert summary['base_training_steps'] == (6080 if smoke else 401280)
    assert summary['fit_transitions'] == (4560 if smoke else 316160)
    assert summary['held_transitions'] == (1520 if smoke else 85120)
    assert summary['checks']['passed']
    verify_sources(summary['source_sha256'], Path(__file__).parent)
    path = folder / 'risk_models.pkl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == summary['model_sha256']
    with path.open('rb') as file:
        models = pickle.load(file)
    assert len(models) == 2
    return models, summary


def score_policy(data, dynamics, policy, offset=0):
    return evaluate(data, dynamics,
                    act_batch=lambda obs: np.asarray(policy.act_det(obs)).reshape(-1), air_offset=offset)


def corrected_policy(base, models):
    return FastCorrectedPolicy(base, models)
