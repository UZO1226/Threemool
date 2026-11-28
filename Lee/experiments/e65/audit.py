"""Independently rebuild all nested-grid choices and replay full trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grid_policy import GridPolicy, load_policy
from rate_io import load_base, load_rates
from policy_io import verify_sources
from rate_model import ConstantRate
from quantile_env import FrozenDynamics, evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'candidate', 'baseline', 'correction', 'comparison', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--null-check', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    pair = json.loads((args.comparison / 'summary.json').read_text())
    assert pair['candidate_method'] == 'e65_nested_grid_h4' and not pair['test_evaluated']
    assert pair['training_steps'] == 401280 and not pair['smoke_only']
    verify_sources(pair['source_sha256'], Path(__file__).parent)
    base, config, metrics, hashes = load_base(args.candidate, args.freeze, args.baseline)
    models, fit = load_rates(args.correction, config, 4, False)
    assert hashes == pair['base_checkpoint_hashes']
    assert fit['model_sha256'] == pair['correction_model_sha256']
    path = args.comparison / 'controller_config.json'
    assert digest(path) == pair['controller_config_sha256']
    policy = load_policy(base, models, path)
    dynamics = FrozenDynamics(args.freeze)
    replays = {}
    offsets = [('native', 0.)]
    if pair['candidate_matched'] is not None:
        offsets.append(('matched', pair['matching']['offset']))
    for phase, offset in offsets:
        totals = {'rows': 0, 'volume': 0.}
        def act(obs):
            unchanged = obs.copy()
            original = np.asarray(base.act_det(obs)).reshape(-1)
            steps = [0., -.125, .125, -.25, .25]
            for integer in range(1, 8):
                if integer != 4:
                    steps.extend([-integer / 32, integer / 32])
            candidates = np.clip(original[:, None] + np.asarray(steps), -1, 1)
            features = np.column_stack((np.repeat(obs, 17, axis=0), candidates.reshape(-1)))
            risk = np.clip(np.column_stack([m.predict(features) for m in models]), 0, 1).sum(-1).reshape(-1, 17)
            expected = candidates[np.arange(len(obs)), risk.argmin(-1)]
            actual = policy.act_det(obs)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(obs, unchanged)
            assert np.max(np.abs(actual - original)) <= .25 + 1e-12
            executed = np.clip(actual + offset, -1, 1)
            totals['volume'] += float(((executed + 1) / 2 * 9000 * .25).sum())
            totals['rows'] += len(obs)
            return actual
        summary, rows = evaluate(args.data, dynamics, act_batch=act, air_offset=offset)
        assert json.loads((args.comparison / f'candidate_{phase}.json').read_text()) == {'summary': summary, 'rows': rows}
        assert totals['rows'] == 152000
        np.testing.assert_allclose(totals['volume'] / 1600, summary['air_m3'], rtol=1e-12, atol=1e-8)
        replays[phase] = {'independent_sklearn_actions_exact': True, 'trajectory_replay_exact': True,
                          'executed_air_accounting_exact': True, 'action_rows': totals['rows'], 'offset': offset}
        print('INDEPENDENT_GRID_REPLAY', config['seed'], phase, flush=True)
    if args.null_check:
        null = GridPolicy(base, [ConstantRate(.3), ConstantRate(.8)])
        summary, _ = evaluate(args.data, dynamics, act_batch=null.act_det)
        for key, value in metrics['val_final'].items():
            np.testing.assert_allclose(summary[key], value, rtol=1e-10, atol=1e-10)
    result = {'passed': True, 'policy_seed': config['seed'], 'replays': replays,
              'flat_risk_null_full_native_exact': args.null_check,
              'comparison_summary_sha256': digest(args.comparison / 'summary.json'),
              'audit_source_sha256': digest(Path(__file__)), 'test_evaluated': False}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('GRID_AUDIT_COMPLETE', config['seed'], flush=True)


if __name__ == '__main__':
    main()
