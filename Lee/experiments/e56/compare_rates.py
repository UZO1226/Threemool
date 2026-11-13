"""Reuse exact e55 replay/matching operations with bounded regression models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

from rate_io import load_base, load_rates, load_reference, file_hashes, method, ROOT
from rate_model import RatePolicy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--horizon', type=int, choices=(1, 4), required=True)
    known, remaining = parser.parse_known_args()
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument('--out', type=Path, required=True)
    paths.add_argument('--candidate', type=Path, required=True)
    target, _ = paths.parse_known_args(remaining)
    if '--null-check' in remaining:
        raise ValueError('Use audit_rates.py for regression-specific null/action checks')
    source = ROOT / 'Lee/experiments/e55/compare.py'
    spec = importlib.util.spec_from_file_location('e55_regression_comparison', source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    def reference(run, freeze, smoke=False):
        return load_reference(run, freeze, False)
    def captured(run, freeze, baseline, smoke=False):
        return load_base(run, freeze, baseline)
    def models(folder, config, smoke=False):
        fitted, summary = load_rates(folder, config, known.horizon, smoke)
        assert summary['loss'] == 'squared_error' and summary['no3_retained']
        assert summary['outputs_clipped'] == [0, 1]
        assert summary['fit_updates'] == ([1] if smoke else list(range(1, 53)))
        assert summary['held_updates'] == ([2] if smoke else list(range(53, 67)))
        assert summary['base_checkpoint_hashes'] == file_hashes(target.candidate / 'model_final')
        return fitted, summary
    with patch.object(legacy, 'load_reference', reference), \
            patch.object(legacy, 'load_captured', captured), \
            patch.object(legacy, 'load_correction', models), \
            patch.object(legacy, 'corrected_policy', RatePolicy), \
            patch.object(sys, 'argv', [str(source), *remaining]):
        legacy.main()
    path = target.out / 'summary.json'
    summary = json.loads(path.read_text())
    summary.update({'candidate_method': method(known.horizon), 'correction_horizon': known.horizon,
                    'loss_matched_regression_control': known.horizon == 1,
                    'additional_ppo_training': False, 'extra_environment_training_steps': 0,
                    'base_training_steps': 401280,
                    'target_mean_rate_not_any_event_probability': True,
                    'comparison_sources_sha256': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                                  for p in Path(__file__).parent.glob('*.py')},
                    'reused_comparison_source_sha256': hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest()})
    path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('RATE_PAIR_COMPLETE', summary['policy_seed'], known.horizon, flush=True)


if __name__ == '__main__':
    main()
