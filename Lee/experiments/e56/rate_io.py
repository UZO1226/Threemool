"""Preserve legacy base-policy guards and validate the new cost-rate artifacts."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / 'Lee/experiments/e55'))
from policy_io import load_captured, load_reference, verify_sources, file_hashes, score_policy
from quantile_env import FrozenDynamics, frozen_digest
from rate_model import RatePolicy


def load_rates(folder, config, horizon, smoke=False):
    summary = json.loads((folder / 'fit_summary.json').read_text())
    assert summary['horizon'] == horizon and horizon in (1, 4)
    assert summary['policy_seed'] == config['seed']
    assert summary['smoke_only'] == smoke and not summary['test_evaluated']
    assert summary['base_training_steps'] == 401280
    assert summary['extra_environment_training_steps'] == 0
    assert summary['fit_transitions'] == (6080 if smoke else 316160)
    assert summary['held_transitions'] == (6080 if smoke else 85120)
    assert summary['checks']['passed']
    verify_sources(summary['source_sha256'], Path(__file__).parent)
    verify_sources(summary['shared_source_sha256'], ROOT)
    path = folder / 'risk_models.pkl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == summary['model_sha256']
    with path.open('rb') as file:
        models = pickle.load(file)
    assert len(models) == 2
    return models, summary


def load_base(run, freeze, baseline):
    return load_captured(run, freeze, baseline, False)


def method(horizon):
    assert horizon in (1, 4)
    return f'ppolag_fcsrl_rate_h{horizon}'
