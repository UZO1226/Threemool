# Adapted from Lee/experiments/e65/preflight.py; e67 changes only the fixed radius/controller and safe status-file retries.
"""Authenticate and compare nested-grid choices on held training states."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from radius_policy import ROOT, OFFSETS, RadiusPolicy, TrustPolicy, public_features, load_policy
from rate_io import load_base, load_rates
from rate_model import slow_rates
from streams import read_stream
from policy_io import verify_sources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'policies', 'baselines', 'rates', 'certificate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    certificate = json.loads(args.certificate.read_text())
    assert certificate['passed'] and certificate['policy_seeds'] == list(range(5))
    assert certificate['data_sha256'] == digest(args.data)
    verify_sources({'authenticate_baseline.py': certificate['source_sha256']},
                   ROOT / 'Lee/experiments/e64')
    args.out.mkdir(parents=True)
    records = []
    for seed in range(5):
        run = args.policies / f'seed_{seed}'
        base, config, _, hashes = load_base(run, args.freeze, args.baselines / f'seed_{seed}')
        models, fit = load_rates(args.rates / f'models/h4/seed_{seed}', config, 4, False)
        reference = certificate['records'][seed]
        assert hashes == fit['base_checkpoint_hashes'] == reference['base_checkpoint_hashes']
        assert fit['model_sha256'] == reference['rate_model_sha256']
        _, held, provenance = read_stream(run, args.data, 4, False)
        obs = held['x'][:, :11]
        assert len(obs) == 85120
        policy, old = RadiusPolicy(base, models), TrustPolicy(base, models, 0)
        config_path = args.out / f'controller_seed_{seed}.json'
        policy.save_config(config_path)
        policy = load_policy(base, models, config_path)
        changed = 0
        gains = []
        for start in range(0, len(obs), 2048):
            batch = obs[start:start + 2048]
            original = np.asarray(base.act_det(batch)).reshape(-1)
            before, after = old.act_det(batch), policy.act_det(batch)
            candidates = np.clip(original[:, None] + np.asarray(OFFSETS), -1, 1)
            assert np.all(np.any(candidates == after[:, None], axis=1))
            assert np.all(np.abs(after - original) <= .5 + 1e-15)
            before_risk = policy.risk.predict(public_features(batch, before)).sum(-1)
            after_risk = policy.risk.predict(public_features(batch, after)).sum(-1)
            assert np.all(after_risk <= before_risk + 1e-14)
            changed += int(np.count_nonzero(before != after))
            gains.extend((before_risk - after_risk).tolist())
        subset = obs[np.linspace(0, len(obs) - 1, 1024, dtype=int)]
        original = np.asarray(base.act_det(subset)).reshape(-1)
        candidates = np.clip(original[:, None] + np.asarray(OFFSETS), -1, 1)
        x = public_features(np.repeat(subset, len(OFFSETS), axis=0), candidates.reshape(-1))
        slow = slow_rates(models, x)
        fast = policy.risk.predict(x)
        np.testing.assert_allclose(fast, slow, atol=1e-12, rtol=0)
        independent = candidates[np.arange(len(subset)), slow.sum(-1).reshape(-1, len(OFFSETS)).argmin(-1)]
        np.testing.assert_array_equal(independent, policy.act_det(subset))
        row = {'policy_seed': seed, 'held_rows': len(obs), 'changed_action_rows': changed,
               'changed_action_percent': changed / len(obs) * 100,
               'mean_predicted_rate_gain': float(np.mean(gains)),
               'independent_sklearn_rows': len(subset), 'independent_actions_exact': True,
               'max_prediction_error': float(np.max(np.abs(fast - slow))),
               'model_sha256': fit['model_sha256'], 'base_checkpoint_hashes': hashes,
               'controller_config_sha256': digest(config_path),
               'capture_provenance': provenance}
        records.append(row)
        print('PREFLIGHT_SEED', seed, changed, flush=True)
    result = {'passed': True, 'policy_seeds': list(range(5)), 'records': records,
              'proceed_to_environment_comparison': any(r['changed_action_rows'] for r in records),
              'new_environment_queries': 0, 'test_evaluated': False, 'new_model_fitting': False,
              'baseline_certificate_sha256': digest(args.certificate), 'data_sha256': digest(args.data),
              'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'scope': 'Held training-state action geometry only, not counterfactual quality.'}
    (args.out / 'preflight.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

