"""Independently rebuild filtering and replay native plus matched executed feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from air_filter import AirFilter
from filter_io import load_base, file_hashes, FrozenDynamics, frozen_digest
from evaluate_filter import score


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'baseline', 'freeze', 'data', 'comparison', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    pair = json.loads((args.comparison / 'summary.json').read_text())
    assert pair['candidate_method'] == 'ppolag_fcsrl_air_filter'
    assert pair['training_steps'] == 401280 and not pair['test_evaluated']
    assert pair['smoke_only'] == args.native_only
    assert pair['freeze_canonical_sha256'] == frozen_digest(args.freeze)
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == pair['data_sha256']
    config_path = args.comparison / 'filter_config.json'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == pair['filter_config_sha256']
    for name, digest in pair['source_sha256'].items():
        assert frozen_digest(Path(__file__).with_name(name)) == digest
    base, config, _, hashes = load_base(args.run, args.freeze, args.baseline)
    assert config['seed'] == pair['policy_seed'] and hashes == pair['candidate_checkpoint_hashes'] == pair['baseline_checkpoint_hashes']
    original = AirFilter.load_config(base, config_path)
    assert original.gain == .5
    checked = [0]
    class Rebuilt:
        def act_det(self, obs):
            b = np.asarray(base.act_det(obs)).reshape(-1).astype(float)
            # Clip the public dose ratio first, independently expressing the
            # same physical conversion and convex average.
            ratio = np.minimum(1., np.maximum(0., obs[:, 10].astype(float)))
            previous = 2 * ratio - 1
            expected = np.minimum(1., np.maximum(-1., (b + previous) / 2))
            np.testing.assert_array_equal(expected, original.act_det(obs))
            checked[0] += len(obs)
            return expected
    dynamics = FrozenDynamics(args.freeze)
    modes = [('native', 0., 'candidate_native.json')]
    if not args.native_only:
        assert pair['matching']['matched_within_one_percent']
        modes.append(('matched', pair['matching']['offset'], 'candidate_matched.json'))
    results = []
    native_payload = None
    for label, offset, filename in modes:
        stored = json.loads((args.comparison / filename).read_text())
        if label == 'matched' and offset == 0.:
            assert stored == native_payload
            results.append({'mode': label, 'offset': offset, 'zero_offset_alias_verified': True})
            continue
        before = checked[0]
        summary, rows, commands = score(args.data, dynamics, Rebuilt(), offset)
        assert summary == stored['summary'] and rows == stored['rows'] and commands == stored['commands']
        if label == 'native':
            native_payload = stored
        results.append({'mode': label, 'offset': offset, 'summary_rows_commands_exact': True,
                        'independent_action_rows': checked[0] - before})
    assert file_hashes(args.run / 'model_final') == hashes
    result = {'passed': True, 'policy_seed': config['seed'], 'smoke_only': args.native_only,
              'test_evaluated': False, 'independent_conversion_and_filter_choices_exact': True,
              'original_nn_observation_unchanged': True, 'base_checkpoint_files_unchanged': True,
              'native_and_matched_replayed': not args.native_only, 'modes': results,
              'scope': 'Independently rebuilt gain.5 actions using current public previous air. Native and, for full trials, matched executed-feedback trajectories and command diagnostics exactly reproduce stored outputs. No training or parameter selection.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('FILTER_ACTION_AUDIT_PASS', config['seed'], json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
