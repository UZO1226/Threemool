"""Construct the unchanged e63 matched controller for one real fresh seed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e63'))
from trust_policy import TrustPolicy, load_policy
from rate_io import load_base, load_rates
from policy_io import score_policy, verify_sources
from quantile_env import FrozenDynamics, frozen_digest
from air_match import match_air


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'baselines', 'rates', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--policy-seed', type=int, choices=range(5, 10), required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    seed = args.policy_seed
    training = json.loads((args.baselines / 'baseline_training_summary.json').read_text())
    rates = json.loads((args.rates / 'rate_preparation_summary.json').read_text())
    assert training['passed'] and rates['passed']
    assert training['policy_seeds'] == rates['policy_seeds'] == list(range(5, 10))
    for folder in (args.baselines, args.rates):
        status = json.loads((folder / 'study_status.json').read_text())
        assert status['phase'] == 'complete' and not status['running'] and not status['pending']
        manifest = json.loads((folder / 'study_manifest.json').read_text())
        verify_sources(manifest['source_sha256'], ROOT)
    rate_record = next(row for row in rates['records'] if row['policy_seed'] == seed)
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    run = args.baselines / f'captured/new_ppolag_fcsrl/seed_{seed}'
    reference = args.baselines / f'original/new_ppolag_fcsrl/seed_{seed}'
    model_folder = args.rates / f'models/h4/seed_{seed}'
    base, config, metrics, hashes = load_base(run, freeze, reference)
    assert config['seed'] == seed
    models, fit = load_rates(model_folder, config, 4, False)
    assert fit['base_checkpoint_hashes'] == hashes == rate_record['base_checkpoint_hashes']
    assert fit['model_sha256'] == rate_record['model_sha256']
    assert sha(data) == fit['data_sha256']
    dynamics = FrozenDynamics(freeze)
    args.out.mkdir(parents=True)
    original, original_rows = score_policy(data, dynamics, base)
    for key, value in metrics['val_final'].items():
        np.testing.assert_allclose(original[key], value, rtol=1e-10, atol=1e-10)
    write(args.out / 'original_fcsrl_native.json', {'summary': original, 'rows': original_rows})
    policy = TrustPolicy(base, models, 0)
    policy.save_config(args.out / 'controller_config.json')
    policy = load_policy(base, models, args.out / 'controller_config.json')
    native, native_rows = score_policy(data, dynamics, policy)
    independent_base, _, _, independent_hashes = load_base(run, freeze, reference)
    independent_models, _ = load_rates(model_folder, config, 4, False)
    assert independent_hashes == hashes
    reloaded = load_policy(independent_base, independent_models, args.out / 'controller_config.json')
    replay, replay_rows = score_policy(data, dynamics, reloaded)
    assert native == replay and native_rows == replay_rows
    write(args.out / 'e63_native.json', {'summary': native, 'rows': native_rows})
    cache = {0.: (native, native_rows)}
    def air_only(offset):
        summary, rows = score_policy(data, dynamics, policy, offset)
        write(args.out / f'query_{len(cache):02d}.json', {'summary': summary, 'rows': rows})
        cache[offset] = summary, rows
        print('REFERENCE_AIR_QUERY', seed, offset, summary['air_m3'], flush=True)
        return summary['air_m3']
    matching = match_air(air_only, original['air_m3'], native['air_m3'])
    offset = matching['offset']
    if offset not in cache:
        air_only(offset)
    matched, matched_rows = cache[offset]
    np.testing.assert_allclose(matched['air_m3'], matching['achieved_air_m3'], rtol=1e-10, atol=1e-8)
    write(args.out / 'e63_matched.json', {'summary': matched, 'rows': matched_rows})
    result = {'policy_seed': seed, 'method': 'e63_matched_risk_only_h4',
        'original_fcsrl_native': original, 'e63_native': native, 'e63_matched': matched,
        'matching': matching, 'base_checkpoint_hashes': hashes, 'rate_model_sha256': fit['model_sha256'],
        'controller_config_sha256': sha(args.out / 'controller_config.json'),
        'base_training_steps': 401280, 'extra_environment_training_steps': 0,
        'test_evaluated': False, 'candidate_evaluation': False, 'checkpoint_replay_verified': True,
        'freeze_canonical_sha256': frozen_digest(freeze), 'data_sha256': sha(data),
        'training_summary_sha256': sha(args.baselines / 'baseline_training_summary.json'),
        'rate_summary_sha256': sha(args.rates / 'rate_preparation_summary.json'),
        'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'scope': 'Fresh e63 comparator only; original and fresh seed blocks remain separate.'}
    write(args.out / 'summary.json', result)
    print('FRESH_E63_REFERENCE_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    main()
