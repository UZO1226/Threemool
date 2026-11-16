"""Independently load final policies and replay their native validation metrics."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from policy_io import load_run, score_policy, file_hashes
from quantile_env import FrozenDynamics, frozen_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'run', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    os.environ['DARROW_FAST_DYNAMICS'] = '1'
    agent, config, metrics, hashes = load_run(args.run, args.freeze, args.allow_smoke)
    summary, rows = score_policy(args.data, FrozenDynamics(args.freeze), agent)
    for key, value in metrics['val_final'].items():
        np.testing.assert_allclose(summary[key], value, rtol=1e-10, atol=1e-10)
    assert file_hashes(args.run / 'model_final') == hashes
    result = {'summary': summary, 'rows': rows, 'checkpoint_hashes': hashes,
        'policy_seed': config['seed'], 'method': config['method'], 'evaluation_environment': 'new',
        'freeze_canonical_sha256': frozen_digest(args.freeze), 'split': 'val', 'test_evaluated': False,
        'smoke_only': args.allow_smoke, 'training_steps': metrics['steps'],
        'checkpoint_loaded_and_evaluated': True, 'training_native_replay_exact_tolerance': 1e-10}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('NATIVE_REPLAY_PASS', config['method'], json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
