"""Inspect fixed proposal curves on held training states without new simulator labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e56'))
from rate_io import load_base, load_rates
from rate_model import FastRate, RatePolicy
from streams import read_stream


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'captures', 'baselines', 'data', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    rows = []
    for seed in range(5):
        run = args.captures / f'seed_{seed}'
        base, config, _, hashes = load_base(run, args.freeze, args.baselines / f'seed_{seed}')
        _, held, provenance = read_stream(run, args.data, 1)
        obs = held['x'][:, :11]
        original = np.asarray(base.act_det(obs)).reshape(-1)
        candidates = np.clip(original[:, None] + [0, -.125, .125, -.25, .25], -1, 1)
        x = np.column_stack((np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
        methods = []
        for horizon in (1, 4):
            models, summary = load_rates(args.study / f'models/h{horizon}/seed_{seed}', config, horizon)
            assert summary['base_checkpoint_hashes'] == hashes
            assert summary['capture_file_sha256'] == provenance['capture_file_sha256']
            risk = FastRate(models).predict(x).sum(-1).reshape(-1, 5)
            best_risk = np.argmin(risk, axis=1)
            score = risk + np.square(candidates - original[:, None])
            selected = np.argmin(score, axis=1)
            risk_action = candidates[np.arange(len(obs)), best_risk]
            actual = candidates[np.arange(len(obs)), selected]
            np.testing.assert_array_equal(actual, RatePolicy(base, models).act_det(obs))
            changed_without_penalty = risk_action != original
            changed_with_penalty = actual != original
            suppressed = changed_without_penalty & ~changed_with_penalty
            gain = risk[:, 0] - risk[np.arange(len(obs)), best_risk]
            assert np.all(gain >= -1e-12)
            methods.append({'horizon': horizon, 'risk_model_sha256': summary['model_sha256'],
                            'held_rows': len(obs),
                            'flat_predicted_risk_fraction_tolerance_1e_minus12': float(np.mean(np.ptp(risk, axis=1) <= 1e-12)),
                            'risk_only_changed_action_fraction': float(changed_without_penalty.mean()),
                            'fixed_penalty_changed_action_fraction': float(changed_with_penalty.mean()),
                            'risk_change_suppressed_by_penalty_fraction': float(suppressed.mean()),
                            'risk_only_choice_counts': np.bincount(best_risk, minlength=5).tolist(),
                            'fixed_penalty_choice_counts': np.bincount(selected, minlength=5).tolist(),
                            'best_predicted_rate_gain_mean': float(gain.mean()),
                            'best_predicted_rate_gain_quantiles_50_90_99': np.quantile(gain, [.5, .9, .99]).tolist(),
                            'selected_predicted_rate_gain_mean': float((risk[:, 0] - risk[np.arange(len(obs)), selected]).mean())})
        rows.append({'policy_seed': seed, 'methods': methods})
        print('HELD_PROPOSAL_DIAGNOSTIC', seed, flush=True)
    result = {'policy_seeds': list(range(5)), 'test_evaluated': False,
              'new_environment_queries': 0, 'controller_parameters_changed': False,
              'states': 'Same last14 captured PPO-training updates, not validation or candidate native trajectories.',
              'methods': rows,
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'scope': 'Exploratory within-model proposal geometry only. Risk-only choices are not evaluated controllers and predicted gains are not causal or measured safety gains. Different target horizons do not make these scores independent performance comparisons.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('RATE_PROPOSAL_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
