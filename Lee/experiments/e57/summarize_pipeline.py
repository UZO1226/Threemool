"""Require identity, serialization and independent primary native replay checks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    preflight = json.loads((here / 'preflight.json').read_text())
    pair = json.loads((args.smoke / 'summary.json').read_text())
    audit = json.loads((args.smoke / 'action_audit.json').read_text())
    assert preflight['passed'] and not preflight['test_evaluated']
    assert pair['smoke_only'] and audit['smoke_only'] and not pair['test_evaluated']
    assert pair['filter_gain'] == .5 and pair['training_steps'] == 401280
    assert pair['candidate_method'] == 'ppolag_fcsrl_air_filter'
    assert pair['candidate_checkpoint_hashes'] == pair['baseline_checkpoint_hashes'] == preflight['base_checkpoint_hashes']
    assert all(pair['checks'].values()) and pair['checkpoint_replay_verified']
    assert audit['passed'] and audit['independent_conversion_and_filter_choices_exact']
    assert audit['modes'][0]['independent_action_rows'] == 152000
    assert pair['matching'] is None and pair['candidate_matched'] is None
    assert pair['extra_environment_training_steps'] == 0
    for mapping in (preflight['source_sha256'], pair['source_sha256']):
        for name, digest in mapping.items():
            assert hashlib.sha256((here / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
    result = {'passed': True, 'smoke_only': True, 'policy_seed': 0,
              'base_training_steps': 401280, 'extra_environment_training_steps': 0,
              'test_evaluated': False, 'original_gain1_native_replay_passed': True,
              'gain1_dtype_nonzero_offset_reset_terminal_checks_passed': True,
              'serialized_gain_half_native_replay_passed': True,
              'independent_gain_half_native_action_rows': 152000,
              'executed_volume_reconstruction_passed': True,
              'matching_not_exercised_in_native_only_smoke': True,
              'shared_matcher_source_sha256': hashlib.sha256((here.parents[1] / 'experiments/e44/air_match.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'runtime_sources': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                  for p in here.glob('*.py')},
              'full_policy_seed_comparisons_completed': [],
              'scope': 'Original identity replay and one full-budget seed native verification. Native primary quality is not an adoption result. Full five-seed air matching and independent matched-feedback replays remain required. Batch orchestration is compiled/source pinned.'}
    args.out.write_text(json.dumps(result, indent=2))
    print('E57_PIPELINE_CHECK_PASS', flush=True)


if __name__ == '__main__':
    main()
