"""Independent sklearn action replay for native and matched risk-only policies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trust_policy import TrustPolicy, load_policy
from compare_trust import digest, verify_preflight, guarded_models
from rate_io import load_base, FrozenDynamics
from rate_model import ConstantRate
from quantile_env import evaluate


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'baseline', 'candidate', 'correction', 'comparison', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path, default=Path(__file__).parent / 'preflight.json')
    parser.add_argument('--null-check', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    pair = json.loads((args.comparison / 'summary.json').read_text())
    assert pair['candidate_method'] == 'ppolag_fcsrl_risk_only_h4'
    assert pair['training_steps'] == 401280 and not pair['test_evaluated'] and not pair['smoke_only']
    assert pair['penalty'] == 0 and pair['correction_horizon'] == 4
    assert pair['preflight_sha256'] == digest(args.preflight)
    for name, expected in pair['comparison_sources_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    preflight = verify_preflight(args.preflight)
    assert preflight['data_sha256'] == digest(args.data)
    base, config, _, hashes = load_base(args.candidate, args.freeze, args.baseline)
    models, fit = guarded_models(args.correction, config, args.candidate, preflight)
    assert hashes == pair['candidate_checkpoint_hashes'] and fit['model_sha256'] == pair['correction_model_sha256']
    path = args.comparison / 'controller_config.json'
    assert digest(path) == pair['controller_config_sha256']
    policy = load_policy(base, models, path)
    assert policy.config['penalty'] == 0
    dynamics = FrozenDynamics(args.freeze)
    replays = {}
    offsets = [('native', 0.)]
    if pair['candidate_matched'] is not None:
        offsets.append(('matched', pair['matching']['offset']))
    for phase, offset in offsets:
        if phase == 'matched' and offset == 0:
            assert json.loads((args.comparison / 'candidate_matched.json').read_text()) == json.loads((args.comparison / 'candidate_native.json').read_text())
            replays[phase] = {**replays['native'], 'same_zero_offset_as_native': True}
            continue
        counts = np.zeros(5, dtype=np.int64)
        totals = {'rows': 0, 'changed': 0, 'executed_m3': 0., 'predicted_rate_change': 0.}

        def act(obs):
            unchanged = obs.copy()
            original = np.asarray(base.act_det(obs)).reshape(-1)
            # Rebuild public features and original candidate order independently.
            candidates = np.clip(original[:, None] + [0, -.125, .125, -.25, .25], -1, 1)
            features = np.column_stack((np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
            rates = np.clip(np.column_stack([model.predict(features) for model in models]), 0, 1)
            risk = rates.sum(-1).reshape(-1, 5)
            selected = np.argmin(risk, axis=1)
            expected = candidates[np.arange(len(obs)), selected]
            actual = policy.act_det(obs)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(obs, unchanged)
            assert np.max(np.abs(actual - original)) <= .25 + 1e-12
            counts[:] += np.bincount(selected, minlength=5)
            totals['rows'] += len(obs)
            totals['changed'] += int(np.count_nonzero(actual != original))
            executed = np.clip(actual + offset, -1, 1)
            totals['executed_m3'] += float(np.sum((executed + 1) / 2 * 9000) * .25)
            totals['predicted_rate_change'] += float((risk[np.arange(len(obs)), selected] - risk[:, 0]).sum())
            return actual

        summary, rows = evaluate(args.data, dynamics, act_batch=act, air_offset=offset)
        saved = json.loads((args.comparison / f'candidate_{phase}.json').read_text())
        assert summary == saved['summary'] and rows == saved['rows']
        assert totals['rows'] == 5 * 8 * 40 * 95
        np.testing.assert_allclose(totals['executed_m3'] / (5 * 8 * 40), summary['air_m3'], rtol=1e-12, atol=1e-8)
        replays[phase] = {'independent_sklearn_actions_exact': True, 'trajectory_replay_exact': True,
            'executed_air_accounting_exact': True, 'offset': offset, 'action_rows': totals['rows'],
            'choice_counts': counts.tolist(), 'changed_action_fraction': totals['changed'] / totals['rows'],
            'mean_predicted_rate_change': totals['predicted_rate_change'] / totals['rows']}
        print('INDEPENDENT_TRUST_REPLAY', phase, flush=True)
    if args.null_check:
        null = TrustPolicy(base, [ConstantRate(.3), ConstantRate(.8)], 0)
        summary, rows = evaluate(args.data, dynamics, act_batch=null.act_det)
        saved = json.loads((args.comparison / 'baseline_native.json').read_text())
        assert summary == saved['summary'] and rows == saved['rows']
    result = {'passed': True, 'policy_seed': config['seed'], 'replays': replays,
              'flat_risk_null_full_native_exact': args.null_check, 'test_evaluated': False,
              'model_refit': False, 'comparison_summary_sha256': digest(args.comparison / 'summary.json'),
              'audit_source_sha256': digest(Path(__file__))}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('TRUST_AUDIT_COMPLETE', config['seed'], flush=True)


if __name__ == '__main__':
    main()
