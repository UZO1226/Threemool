"""Verify fixed risk-only and original-penalty decisions on held training states."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from trust_policy import ROOT, OFFSETS, TrustPolicy, load_policy, configuration
from rate_model import ConstantRate, RatePolicy, slow_rates
from rate_io import load_base, load_rates
from streams import read_stream


class BoundaryBase:
    def act_det(self, obs):
        return np.asarray(obs)[:, 0].copy()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'captures', 'baselines', 'study', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    diagnostic = json.loads((ROOT / 'Lee/experiments/e56/proposal_diagnostic.json').read_text())
    synthetic = np.zeros((7, 11))
    synthetic[:, 0] = [-1, -.999, -.75, 0, .75, .999, 1]
    saved = synthetic.copy()
    null_models = [ConstantRate(.3), ConstantRate(.8)]
    base = BoundaryBase()
    for penalty in (0, 1):
        policy = TrustPolicy(base, null_models, penalty)
        np.testing.assert_array_equal(policy.act_det(synthetic), synthetic[:, 0])
    np.testing.assert_array_equal(synthetic, saved)
    for penalty in (-1, .1, 2, True, float('nan')):
        try:
            configuration(penalty)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported penalty accepted')
    with tempfile.TemporaryDirectory(prefix='e63_config_') as folder:
        path = Path(folder) / 'config.json'
        policy = TrustPolicy(base, null_models)
        policy.save_config(path)
        np.testing.assert_array_equal(load_policy(base, null_models, path).act_det(synthetic), policy.act_det(synthetic))
        bad = json.loads(path.read_text())
        bad['candidate_offsets'][-1] = .5
        path.write_text(json.dumps(bad))
        try:
            load_policy(base, null_models, path)
        except ValueError:
            pass
        else:
            raise AssertionError('Altered configuration accepted')
    rows = []
    for seed in range(5):
        run = args.captures / f'seed_{seed}'
        base, config, _, hashes = load_base(run, args.freeze, args.baselines / f'seed_{seed}')
        models, fit = load_rates(args.study / f'models/h4/seed_{seed}', config, 4)
        assert fit['base_checkpoint_hashes'] == hashes
        _, held, provenance = read_stream(run, args.data, 4)
        assert fit['capture_file_sha256'] == provenance['capture_file_sha256']
        obs = held['x'][:, :11]
        assert len(obs) == 85120
        untouched = obs.copy()
        original = np.asarray(base.act_det(obs)).reshape(-1)
        candidates = np.clip(original[:, None] + np.array(OFFSETS), -1, 1)
        x = np.column_stack((np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
        rates = slow_rates(models, x)
        risk = rates.sum(-1).reshape(-1, 5)
        expected_index = np.argmin(risk, axis=1)
        expected = candidates[np.arange(len(obs)), expected_index]
        primary = TrustPolicy(base, models)
        actual = primary.act_det(obs)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(TrustPolicy(base, models, 1).act_det(obs), RatePolicy(base, models).act_det(obs))
        np.testing.assert_array_equal(obs, untouched)
        assert np.max(np.abs(actual - original)) <= .25 + 1e-12
        old = next(record for record in diagnostic['methods'] if record['policy_seed'] == seed)
        previous = next(record for record in old['methods'] if record['horizon'] == 4)
        counts = np.bincount(expected_index, minlength=5).tolist()
        assert counts == previous['risk_only_choice_counts']
        np.testing.assert_allclose(np.mean(actual != original), previous['risk_only_changed_action_fraction'], rtol=1e-12)
        rows.append({'policy_seed': seed, 'held_rows': len(obs), 'model_sha256': fit['model_sha256'],
                     'base_checkpoint_hashes': hashes, 'risk_only_choice_counts': counts,
                     'changed_action_fraction': float(np.mean(actual != original)),
                     'original_penalty_identity_exact': True, 'independent_sklearn_actions_exact': True})
        print('HELD_PREFLIGHT_PASS', seed, flush=True)
    result = {'passed': True, 'policy_seeds': list(range(5)), 'records': rows,
              'flat_risk_boundary_identity': True, 'unchanged_inputs': True,
              'configuration_roundtrip_and_tamper_rejection': True, 'unsupported_penalties_rejected': True,
              'extra_environment_queries': 0, 'extra_ppo_training': False, 'model_refit': False,
              'validation_evaluated': False, 'test_evaluated': False,
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
