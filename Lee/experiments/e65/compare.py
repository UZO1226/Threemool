"""Compare the fixed nested grid against the authenticated matched e63 policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grid_policy import ROOT, GridPolicy, TrustPolicy, load_policy
from rate_io import load_base, load_rates
from policy_io import score_policy, verify_sources
from quantile_env import FrozenDynamics, frozen_digest
from air_match import match_air


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'candidate', 'baseline', 'correction',
                 'certificate', 'preflight', 'previous', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    cert = json.loads(args.certificate.read_text())
    preflight = json.loads(args.preflight.read_text())
    assert cert['passed'] and preflight['passed'] and preflight['proceed_to_environment_comparison']
    assert preflight['baseline_certificate_sha256'] == digest(args.certificate)
    assert preflight['data_sha256'] == cert['data_sha256'] == digest(args.data)
    verify_sources(preflight['source_sha256'], Path(__file__).parent)
    base, config, _, hashes = load_base(args.candidate, args.freeze, args.baseline)
    models, fit = load_rates(args.correction, config, 4, False)
    seed = config['seed']
    reference = cert['records'][seed]
    assert reference['policy_seed'] == seed
    assert hashes == reference['base_checkpoint_hashes'] == fit['base_checkpoint_hashes']
    assert fit['model_sha256'] == reference['rate_model_sha256']
    assert frozen_digest(args.freeze) == cert['freeze_canonical_sha256']
    args.out.mkdir(parents=True)
    dynamics = FrozenDynamics(args.freeze)
    old = TrustPolicy(base, models, 0)
    baseline, baseline_rows = score_policy(args.data, dynamics, old, reference['fixed_e63_offset_after_wrapper'])
    saved = args.previous / f'comparisons/seed_{seed}/candidate_matched.json'
    assert digest(saved) == reference['matched_file_sha256']
    assert json.loads(saved.read_text()) == {'summary': baseline, 'rows': baseline_rows}
    assert baseline == reference['primary_reference_metrics']
    write(args.out / 'baseline_native.json', {'summary': baseline, 'rows': baseline_rows})
    policy = GridPolicy(base, models)
    policy.save_config(args.out / 'controller_config.json')
    policy = load_policy(base, models, args.out / 'controller_config.json')
    native, native_rows = score_policy(args.data, dynamics, policy)
    other_base, _, _, _ = load_base(args.candidate, args.freeze, args.baseline)
    other_models, _ = load_rates(args.correction, config, 4, False)
    other = load_policy(other_base, other_models, args.out / 'controller_config.json')
    replay, replay_rows = score_policy(args.data, dynamics, other)
    assert native == replay and native_rows == replay_rows
    write(args.out / 'candidate_native.json', {'summary': native, 'rows': native_rows})
    matched = matching = None
    if not args.native_only:
        cache = {0.: (native, native_rows)}
        def air_only(offset):
            summary, rows = score_policy(args.data, dynamics, policy, offset)
            write(args.out / f'query_{len(cache):02d}.json', {'summary': summary, 'rows': rows})
            cache[offset] = summary, rows
            print('AIR_QUERY', seed, offset, summary['air_m3'], flush=True)
            return summary['air_m3']
        matching = match_air(air_only, baseline['air_m3'], native['air_m3'])
        offset = matching['offset']
        if offset not in cache:
            air_only(offset)
        matched, rows = cache[offset]
        np.testing.assert_allclose(matched['air_m3'], matching['achieved_air_m3'], rtol=1e-10, atol=1e-8)
        write(args.out / 'candidate_matched.json', {'summary': matched, 'rows': rows})
    result = {'policy_seed': seed, 'kind': 'agent', 'evaluation_environment': 'new',
              'baseline_method': 'e63_matched_risk_only_h4', 'candidate_method': 'e65_nested_grid_h4',
              'baseline_training_environment': 'new', 'candidate_training_environment': 'new',
              'baseline_native': baseline, 'candidate_native': native, 'candidate_matched': matched,
              'original_fcsrl_reference': reference['original_fcsrl_metrics'],
              'baseline_fixed_offset': reference['fixed_e63_offset_after_wrapper'],
              'matching': matching, 'freeze_canonical_sha256': frozen_digest(args.freeze),
              'base_checkpoint_hashes': hashes, 'correction_model_sha256': fit['model_sha256'],
              'training_steps': 401280, 'test_evaluated': False, 'smoke_only': False,
              'pipeline_smoke_only': args.native_only, 'checkpoint_replay_verified': True,
              'new_model_fitting': False, 'extra_environment_training_steps': 0,
              'baseline_certificate_sha256': digest(args.certificate), 'preflight_sha256': digest(args.preflight),
              'controller_config_sha256': digest(args.out / 'controller_config.json'),
              'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'scope': 'Exploratory reused validation; baseline_native denotes authenticated matched e63.'}
    write(args.out / 'summary.json', result)
    print('GRID_COMPARISON_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    main()
