"""Describe pre-action violation-regime exposure without fitting or rollouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'diagnostic', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    results = args.workspace.resolve() / 'results'
    preflight_path = results / 'e69_projection_fit/preflight.json'
    preflight = json.loads(preflight_path.read_text())
    native = json.loads(args.diagnostic.read_text())
    assert preflight['passed'] and native['passed']
    assert native['policy_seeds'] == preflight['policy_seeds'] == list(range(5))
    records = []
    def summarize(counts):
        total = int(sum(counts))
        return {'rows': total, 'regime_counts': list(map(int, counts)),
            'current_nh4_exposure_percent': float((counts[2] + counts[3]) / total * 100),
            'current_do_exposure_percent': float((counts[1] + counts[3]) / total * 100)}
    for seed in range(5):
        fit = next(row for row in preflight['records'] if row['policy_seed'] == seed)
        record = next(row for row in native['records'] if row['policy_seed'] == seed)
        assert fit['base_checkpoint_hashes'] == record['base_checkpoint_hashes']
        counts = {'fit': np.zeros(4, dtype=np.int64), 'held': np.zeros(4, dtype=np.int64)}
        for update in range(1, 67):
            path = results / f'e55_study/policies/new_ppolag_fcsrl/seed_{seed}/transitions/update_{update:03d}.npz'
            assert sha(path) == fit['capture_provenance']['capture_file_sha256'][path.name]
            with np.load(path, allow_pickle=False) as saved:
                obs = saved['obs'].reshape(-1, 11)
                assert obs.shape == (6080, 11) and np.isfinite(obs).all()
                regime = 2 * (obs[:, 0] > 4).astype(int) + (obs[:, 2:5].max(-1) > 2).astype(int)
                counts['fit' if update <= 52 else 'held'] += np.bincount(regime, minlength=4)
        assert counts['fit'].sum() == 316160 and counts['held'].sum() == 85120
        populations = {name: summarize(value) for name, value in counts.items()}
        for occupancy in record['occupancies']:
            populations[occupancy['control'] + '_native'] = summarize(occupancy['regime_counts'])
        records.append({'policy_seed': seed, 'populations': populations})
    result = {'passed': True, 'policy_seeds': list(range(5)), 'records': records,
        'test_evaluated': False, 'new_environment_queries': 0, 'new_model_fitting': False,
        'candidate_selection': False, 'source_sha256': sha(Path(__file__)),
        'projection_preflight_sha256': sha(preflight_path), 'transport_summary_sha256': sha(args.diagnostic),
        'scope': 'Posthoc descriptive current-state exposure, not next-step event frequency, causal identification or a new candidate result.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps([{name: r['populations'][name]['current_do_exposure_percent']
                       for name in ('fit', 'held', 'e63_native', 'e69_native')} for r in records]), flush=True)


if __name__ == '__main__':
    main()
