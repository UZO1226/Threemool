"""Reuse unchanged baseline/native/air matching with the fixed risk-only policy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

from trust_policy import ROOT, TrustPolicy, load_policy
from rate_io import load_base, load_rates, load_reference, file_hashes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guarded_models(folder, config, candidate, preflight):
    models, summary = load_rates(folder, config, 4, False)
    assert summary['loss'] == 'squared_error' and summary['no3_retained']
    assert summary['outputs_clipped'] == [0, 1]
    assert summary['fit_updates'] == list(range(1, 53))
    assert summary['held_updates'] == list(range(53, 67))
    assert summary['base_checkpoint_hashes'] == file_hashes(candidate / 'model_final')
    stored = next(row for row in preflight['records'] if row['policy_seed'] == config['seed'])
    assert stored['model_sha256'] == summary['model_sha256']
    assert stored['base_checkpoint_hashes'] == summary['base_checkpoint_hashes']
    return models, summary


def verify_preflight(path):
    preflight = json.loads(path.read_text())
    assert preflight['passed'] and preflight['policy_seeds'] == list(range(5))
    assert preflight['extra_environment_queries'] == 0 and not preflight['test_evaluated']
    for name, expected in preflight['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    return preflight


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--preflight', type=Path, default=Path(__file__).parent / 'preflight.json')
    known, remaining = parser.parse_known_args()
    paths = argparse.ArgumentParser(add_help=False)
    for name in ('out', 'candidate', 'data'):
        paths.add_argument('--' + name, type=Path, required=True)
    target, _ = paths.parse_known_args(remaining)
    if '--allow-smoke' in remaining or '--null-check' in remaining:
        raise ValueError('Full base budgets only; run audit_trust.py for independent null/actions')
    preflight = verify_preflight(known.preflight)
    assert digest(target.data) == preflight['data_sha256']
    source = ROOT / 'Lee/experiments/e55/compare.py'
    spec = importlib.util.spec_from_file_location('e63_legacy_comparison', source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    def reference(run, freeze, smoke=False):
        return load_reference(run, freeze, False)

    def captured(run, freeze, baseline, smoke=False):
        return load_base(run, freeze, baseline)

    def models(folder, config, smoke=False):
        return guarded_models(folder, config, target.candidate, preflight)

    def corrected(base, fitted):
        policy = TrustPolicy(base, fitted, 0)
        path = target.out / 'controller_config.json'
        if not path.exists():
            policy.save_config(path)
        restored = load_policy(base, fitted, path)
        assert restored.config['penalty'] == 0
        return restored

    with patch.object(legacy, 'load_reference', reference), \
            patch.object(legacy, 'load_captured', captured), \
            patch.object(legacy, 'load_correction', models), \
            patch.object(legacy, 'corrected_policy', corrected), \
            patch.object(sys, 'argv', [str(source), *remaining]):
        legacy.main()
    path = target.out / 'summary.json'
    summary = json.loads(path.read_text())
    summary.update({'candidate_method': 'ppolag_fcsrl_risk_only_h4',
                    'correction_horizon': 4, 'penalty': 0, 'hard_action_radius': .25,
                    'additional_ppo_training': False, 'extra_environment_training_steps': 0,
                    'base_training_steps': 401280, 'new_model_fitting': False,
                    'pipeline_smoke_only': '--native-only' in remaining,
                    'target_mean_rate_not_any_event_probability': True,
                    'preflight_sha256': digest(known.preflight),
                    'controller_config_sha256': digest(target.out / 'controller_config.json'),
                    'comparison_sources_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                    'reused_comparison_source_sha256': digest(source)})
    path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('TRUST_PAIR_COMPLETE', summary['policy_seed'], flush=True)


if __name__ == '__main__':
    main()
