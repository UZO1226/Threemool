"""Compare the unchanged base policy and fixed correction at matched actual air."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from policy_io import (load_captured, load_correction, load_reference, file_hashes,
                       score_policy, corrected_policy)
from quantile_env import FrozenDynamics, frozen_digest
from air_match import match_air
from cost_model import ConstantProbability, CorrectedPolicy


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'baseline', 'candidate', 'correction', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--null-check', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ['DARROW_FAST_DYNAMICS'] = '1'
    reference, reference_config, reference_metrics, reference_hashes = load_reference(
        args.baseline, args.freeze, args.allow_smoke)
    base, config, metrics, hashes = load_captured(args.candidate, args.freeze, args.baseline, args.allow_smoke)
    assert reference_config['config'] == config['config'] and reference_hashes == hashes
    models, fit = load_correction(args.correction, config, args.allow_smoke)
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == fit['data_sha256']
    dynamics = FrozenDynamics(args.freeze)
    native, base_rows = score_policy(args.data, dynamics, base)
    for key, value in reference_metrics['val_final'].items():
        np.testing.assert_allclose(native[key], value, rtol=1e-10, atol=1e-10)
    assert metrics['val_final'] == reference_metrics['val_final']
    write(args.out / 'baseline_native.json', {'summary': native, 'rows': base_rows})
    policy = corrected_policy(base, models)
    # Independent serialization reload is used for the second complete day replay.
    with (args.correction / 'risk_models.pkl').open('rb') as file:
        restored = pickle.load(file)
    independent_base, _, _, _ = load_captured(args.candidate, args.freeze, args.baseline, args.allow_smoke)
    other = corrected_policy(independent_base, restored)
    candidate, candidate_rows = score_policy(args.data, dynamics, policy)
    replay, replay_rows = score_policy(args.data, dynamics, other)
    assert candidate == replay and candidate_rows == replay_rows
    write(args.out / 'candidate_native.json', {'summary': candidate, 'rows': candidate_rows})
    checks = {'base_checkpoint_files_exact': True, 'base_native_replay': True,
              'corrected_serialized_independent_native_replay_exact': True}
    if args.null_check:
        null = corrected_policy(base, [ConstantProbability(.3), ConstantProbability(.8)])
        null_summary, null_rows = score_policy(args.data, dynamics, null)
        assert null_summary == native and null_rows == base_rows
        # Compare original sklearn and fast decisions on every visited native state.
        slow = CorrectedPolicy(base, models)
        checked = [0]
        def both(obs):
            fast_action = policy.act_det(obs)
            np.testing.assert_array_equal(fast_action, slow.act_det(obs))
            checked[0] += len(obs)
            return fast_action
        from quantile_env import evaluate
        slow_checked, slow_rows = evaluate(args.data, dynamics, act_batch=both)
        assert slow_checked == candidate and slow_rows == candidate_rows
        checks.update({'constant_risk_null_full_day_exact': True,
                       'fast_sklearn_full_day_actions_exact': True,
                       'checked_action_rows': checked[0]})
    matching = matched = None
    if not args.native_only:
        cache = {0.: (candidate, candidate_rows)}
        def air_only(offset):
            summary, rows = score_policy(args.data, dynamics, policy, offset)
            write(args.out / f'query_{len(cache):02d}.json', {'summary': summary, 'rows': rows})
            cache[offset] = summary, rows
            print('AIR_QUERY', offset, summary['air_m3'], flush=True)
            return summary['air_m3']
        matching = match_air(air_only, native['air_m3'], candidate['air_m3'])
        offset = matching['offset']
        if offset not in cache:
            air_only(offset)
        matched, matched_rows = cache[offset]
        np.testing.assert_allclose(matched['air_m3'], matching['achieved_air_m3'], rtol=1e-10, atol=1e-8)
        write(args.out / 'candidate_matched.json', {'summary': matched, 'rows': matched_rows})
    assert file_hashes(args.candidate / 'model_final') == hashes
    assert file_hashes(args.baseline / 'model_final') == reference_hashes
    assert hashlib.sha256((args.correction / 'risk_models.pkl').read_bytes()).hexdigest() == fit['model_sha256']
    result = {'policy_seed': config['seed'], 'kind': 'agent', 'evaluation_environment': 'new',
              'baseline_method': 'ppolag_fcsrl', 'candidate_method': 'ppolag_fcsrl_one_step_correction',
              'baseline_training_environment': 'new', 'candidate_training_environment': 'new',
              'baseline_native': native, 'candidate_native': candidate, 'candidate_matched': matched,
              'matching': matching, 'freeze_canonical_sha256': frozen_digest(args.freeze),
              'baseline_checkpoint_hashes': reference_hashes, 'candidate_checkpoint_hashes': hashes,
              'correction_model_sha256': fit['model_sha256'], 'training_steps': metrics['steps'],
              'test_evaluated': False, 'smoke_only': args.allow_smoke, 'checkpoint_replay_verified': True,
              'checks': checks, 'extra_environment_training_steps': 0,
              'scope': 'Final unchanged base plus fixed learned correction. Offset after correction, air-only matching. Exploratory reused validation, not a plant safety claim.'}
    write(args.out / 'summary.json', result)
    print('CORRECTION_PAIR_COMPLETE', config['seed'], json.dumps({'native': candidate, 'matched': matched}), flush=True)


if __name__ == '__main__':
    main()
