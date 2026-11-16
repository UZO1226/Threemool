"""Replay the unchanged policy and serialized actuator filter at matched actual air."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from air_filter import AirFilter
from filter_io import load_base, load_reference, file_hashes, FrozenDynamics, frozen_digest, ROOT
from evaluate_filter import score
from air_match import match_air


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'baseline', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    _, ref_config, ref_metrics, ref_hashes = load_reference(args.baseline, args.freeze, False)
    base, config, metrics, hashes = load_base(args.candidate, args.freeze, args.baseline)
    assert ref_hashes == hashes and ref_config['config'] == config['config']
    freeze = json.loads(args.freeze.read_text())
    expected = [value for name, value in freeze['frozen_files_sha256'].items() if name.endswith('darrow_15min.pkl')]
    assert len(expected) == 1 and hashlib.sha256(args.data.read_bytes()).hexdigest() == expected[0]
    dynamics = FrozenDynamics(args.freeze)
    baseline, baseline_rows, baseline_commands = score(args.data, dynamics, base)
    for key, value in ref_metrics['val_final'].items():
        np.testing.assert_allclose(baseline[key], value, atol=1e-10, rtol=1e-10)
    assert metrics['val_final'] == ref_metrics['val_final']
    write(args.out / 'baseline_native.json', {'summary': baseline, 'rows': baseline_rows, 'commands': baseline_commands})
    policy = AirFilter(base, .5)
    policy.save_config(args.out / 'filter_config.json')
    config_hash = hashlib.sha256((args.out / 'filter_config.json').read_bytes()).hexdigest()
    candidate, candidate_rows, candidate_commands = score(args.data, dynamics, policy)
    restored_base, _, _, _ = load_base(args.candidate, args.freeze, args.baseline)
    restored = AirFilter.load_config(restored_base, args.out / 'filter_config.json')
    replay, replay_rows, replay_commands = score(args.data, dynamics, restored)
    assert candidate == replay and candidate_rows == replay_rows and candidate_commands == replay_commands
    write(args.out / 'candidate_native.json', {'summary': candidate, 'rows': candidate_rows, 'commands': candidate_commands})
    cache = {0.: (candidate, candidate_rows, candidate_commands)}
    matching = matched = matched_commands = None
    if not args.native_only:
        def air_only(offset):
            evaluated = score(args.data, dynamics, policy, offset)
            write(args.out / f'query_{len(cache):02d}.json',
                  {'summary': evaluated[0], 'rows': evaluated[1], 'commands': evaluated[2]})
            cache[offset] = evaluated
            print('AIR_QUERY', offset, evaluated[0]['air_m3'], flush=True)
            return evaluated[0]['air_m3']
        matching = match_air(air_only, baseline['air_m3'], candidate['air_m3'])
        offset = matching['offset']
        if offset not in cache:
            air_only(offset)
        matched, matched_rows, matched_commands = cache[offset]
        np.testing.assert_allclose(matched['air_m3'], matching['achieved_air_m3'], atol=1e-8, rtol=1e-10)
        write(args.out / 'candidate_matched.json', {'summary': matched, 'rows': matched_rows, 'commands': matched_commands})
    assert file_hashes(args.candidate / 'model_final') == hashes
    assert file_hashes(args.baseline / 'model_final') == ref_hashes
    assert hashlib.sha256((args.out / 'filter_config.json').read_bytes()).hexdigest() == config_hash
    result = {'policy_seed': config['seed'], 'kind': 'agent', 'evaluation_environment': 'new',
              'baseline_method': 'ppolag_fcsrl', 'candidate_method': 'ppolag_fcsrl_air_filter',
              'baseline_training_environment': 'new', 'candidate_training_environment': 'new',
              'baseline_native': baseline, 'candidate_native': candidate, 'candidate_matched': matched,
              'matching': matching, 'baseline_checkpoint_hashes': ref_hashes, 'candidate_checkpoint_hashes': hashes,
              'freeze_canonical_sha256': frozen_digest(args.freeze), 'filter_gain': .5,
              'filter_config_sha256': config_hash, 'training_steps': 401280,
              'data_sha256': expected[0],
              'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                                      'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'test_evaluated': False, 'smoke_only': args.smoke, 'checkpoint_replay_verified': True,
              'additional_ppo_training': False, 'extra_environment_training_steps': 0,
              'checks': {'base_native_replay': True, 'base_checkpoint_files_exact': True,
                         'serialized_primary_native_replay_exact': True, 'executed_air_volume_reconstructed': True},
              'commands': {'baseline_native': baseline_commands, 'candidate_native': candidate_commands,
                           'candidate_matched': matched_commands},
              'source_sha256': {p.name: frozen_digest(p) for p in Path(__file__).parent.glob('*.py')},
              'scope': 'Fixed gain.5 executed-air filter on unchanged FCSRL, offset after filtering. Exploratory reused validation and frozen surrogate; command smoothness is not adoption.'}
    write(args.out / 'summary.json', result)
    print('FILTER_PAIR_COMPLETE', config['seed'], json.dumps({'native': candidate, 'matched': matched}), flush=True)


if __name__ == '__main__':
    main()
