"""Require exact unchanged-reference files before fitting a correction model."""
import argparse
import json
from pathlib import Path

import torch

from policy_io import load_captured, load_reference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('run', 'freeze', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    _, reference, _, reference_hashes = load_reference(args.baseline, args.freeze, args.allow_smoke)
    _, config, metrics, hashes = load_captured(args.run, args.freeze, args.baseline, args.allow_smoke)
    assert hashes == reference_hashes and config['seed'] == reference['seed']
    result = {'passed': True, 'policy_seed': config['seed'], 'steps': metrics['steps'],
              'source_config_and_budget_guards': True, 'all_final_checkpoint_files_exact': True,
              'stored_native_metrics_exact': True, 'checkpoint_hashes': hashes,
              'test_evaluated': False, 'smoke_only': args.allow_smoke,
              'scope': 'Exact file/config/source/native-record equivalence before fitting. Independent native replay is a separate comparison check.'}
    args.out.write_text(json.dumps(result, indent=2))
    print('BASE_EQUIVALENCE_PASS', config['seed'], flush=True)
