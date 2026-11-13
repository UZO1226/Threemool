"""Compare final causal-history variants with unchanged quality-blind air matching."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from policy_io import load_run, score_policy, file_hashes
from quantile_env import FrozenDynamics, frozen_digest
from air_match import match_air


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'baseline', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--baseline-evaluation', type=Path)
    parser.add_argument('--candidate-evaluation', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ['DARROW_FAST_DYNAMICS'] = '1'
    freeze = json.loads(args.freeze.read_text())
    expected = next(value for name, value in freeze['frozen_files_sha256'].items() if name.endswith('darrow_15min.pkl'))
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == expected
    baseline, bc, bm, bh = load_run(args.baseline, args.freeze, args.allow_smoke)
    candidate, cc, cm, ch = load_run(args.candidate, args.freeze, args.allow_smoke)
    assert bc['seed'] == cc['seed'] and bc['config'] == cc['config']
    assert bc['representation'] == cc['representation']
    assert bc['method'] == 'ppolag_fcsrl'
    assert cc['method'] in ('ppolag_fcsrl_history', 'ppolag_fcsrl_repeated')
    assert bc['experiment_metadata']['environment'] == cc['experiment_metadata']['environment'] == 'new'
    dynamics = FrozenDynamics(args.freeze)
    def score(agent, offset, label):
        summary, rows = score_policy(args.data, dynamics, agent, air_offset=offset)
        (args.out / f'{label}.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2, allow_nan=False))
        return summary, rows
    def native(agent, config, metrics, hashes, cached, label):
        if cached is None:
            summary, rows = score(agent, 0, label)
        else:
            stored = json.loads(cached.read_text())
            assert stored['checkpoint_hashes'] == hashes and stored['policy_seed'] == config['seed']
            assert stored['method'] == config['method'] and stored['evaluation_environment'] == 'new'
            assert stored['freeze_canonical_sha256'] == frozen_digest(args.freeze) and stored['split'] == 'val'
            assert not stored['test_evaluated'] and stored['smoke_only'] == args.allow_smoke
            assert stored['training_steps'] == metrics['steps'] and stored['checkpoint_loaded_and_evaluated']
            summary, rows = stored['summary'], stored['rows']
            (args.out / f'{label}.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2))
        for key, value in metrics['val_final'].items():
            np.testing.assert_allclose(summary[key], value, rtol=1e-10, atol=1e-10)
        return summary, rows
    bscore, _ = native(baseline, bc, bm, bh, args.baseline_evaluation, 'baseline_native')
    cscore, crows = native(candidate, cc, cm, ch, args.candidate_evaluation, 'candidate_native')
    matching, matched = None, None
    if not args.native_only:
        cache = {0.: (cscore, crows)}
        def air_only(offset):
            value = score(candidate, offset, f'query_{len(cache):02d}')
            cache[offset] = value
            print('AIR_QUERY', offset, value[0]['air_m3'], flush=True)
            return value[0]['air_m3']
        matching = match_air(air_only, bscore['air_m3'], cscore['air_m3'])
        offset = matching['offset']
        if offset not in cache:
            cache[offset] = score(candidate, offset, 'matched_endpoint')
        matched, rows = cache[offset]
        np.testing.assert_allclose(matched['air_m3'], matching['achieved_air_m3'], rtol=1e-10, atol=1e-8)
        (args.out / 'candidate_matched.json').write_text(json.dumps({'summary': matched, 'rows': rows}, indent=2))
    assert file_hashes(args.baseline / 'model_final') == bh and file_hashes(args.candidate / 'model_final') == ch
    result = {'policy_seed': bc['seed'], 'kind': 'agent', 'evaluation_environment': 'new',
        'baseline_method': bc['method'], 'candidate_method': cc['method'],
        'baseline_training_environment': 'new', 'candidate_training_environment': 'new',
        'baseline_native': bscore, 'candidate_native': cscore, 'candidate_matched': matched, 'matching': matching,
        'freeze_canonical_sha256': frozen_digest(args.freeze), 'baseline_checkpoint_hashes': bh,
        'candidate_checkpoint_hashes': ch, 'training_steps': bm['steps'], 'test_evaluated': False,
        'smoke_only': args.allow_smoke, 'checkpoint_replay_verified': True,
        'scope': 'Exploratory reused validation, final policies, unchanged frozen law. Air-only matching changes executed actions and their observed history.'}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('PAIR_COMPLETE', bc['seed'], cc['method'], json.dumps({'native': cscore, 'matched': matched}), flush=True)


if __name__ == '__main__':
    main()
