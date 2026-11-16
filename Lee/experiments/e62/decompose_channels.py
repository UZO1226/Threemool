"""Calibration-only two-factor decomposition of e61 context PIT channels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e61'))
from context_paths import initial_context, fit_scale, donor_pools, paths
from runtime import q, load_model, rollout
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e61', 'preflight', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads((args.e61 / 'summary.json').read_text())
    preflight = json.loads(args.preflight.read_text())
    assert source['data_sha256'] == preflight['data_sha256'] == digest(args.data)
    assert source['test_evaluated'] is False
    for name, expected in source['source_sha256'].items():
        assert digest(ROOT / 'Lee/experiments/e61' / name) == expected
    assert digest(args.e51 / 'summary.json') == source['parent_summary_sha256']['e51']
    parent = json.loads((args.e51 / 'summary.json').read_text())
    for name, expected in parent['model_hashes'].items():
        assert digest(args.e51 / name) == expected
    raw = pd.read_pickle(args.data)
    fit, cal = source['fit_days'], source['calibration_days']
    assert len(fit) == 158 and len(cal) == 30
    assert set(cal).isdisjoint(fit + raw['days']['val'] + raw['days']['test'])
    batch = q.arrays(raw['data'], cal)
    context = initial_context(batch)
    mean, sd_context, _ = fit_scale(initial_context(q.arrays(raw['data'], fit)))
    np.testing.assert_array_equal(mean, source['fit_context_mean'])
    np.testing.assert_array_equal(sd_context, source['fit_context_sd'])
    bank = np.load(args.e51 / 'pit_bank.npz')['pit']
    assert hashlib.sha256(bank.tobytes()).hexdigest() == preflight['bank_array_sha256']['e51']
    noises = {}
    for mode, k in [('global', None), ('context', 5)]:
        pools = donor_pools(context, context, cal, cal, mean, sd_context, k=k, exclude_self=True)
        noises[mode] = paths(bank, cal, pools)
    dyn = load_model(args.e51, 'e51')
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in fit]).std(0)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'data_sha256': digest(args.data), 'e61_summary_sha256': digest(args.e61 / 'summary.json'),
                'source_sha256': digest(Path(__file__)), 'model_refit': False,
                'validation_evaluated': False, 'test_evaluated': False, 'policy_seeds': [],
                'scope': 'Calibration-only internal-model channel decomposition; mixed noise breaks NH4/DO joint pairing.'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    def job(task):
        nh4_mode, do_mode = task
        noise = noises['global'].copy()
        noise[..., 0] = noises[nh4_mode][..., 0]
        noise[..., 2:] = noises[do_mode][..., 2:]
        trajectory = rollout(dyn, batch, noise)
        score = q.summarize(trajectory, batch[0], sd)
        metrics, errors = event_metrics(trajectory, batch[0])
        if nh4_mode == do_mode:
            suffix = 'global' if nh4_mode == 'global' else 'context5'
            reference = source['calibration_leave_one_day_out']['e51_' + suffix]
            for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
                np.testing.assert_allclose(score[key], reference[key], rtol=1e-12, atol=1e-14)
        key = 'nh4_' + nh4_mode + '_do_' + do_mode
        record = {'score': score, 'event_metrics': metrics}
        (args.out / (key + '.json')).write_text(json.dumps(record, indent=2, allow_nan=False))
        print('COMPLETED', key, flush=True)
        return key, record, errors

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(job, [(n, d) for n in ('global', 'context') for d in ('global', 'context')]))
    records = {key: record for key, record, _ in completed}
    errors = {key: daily for key, _, daily in completed}
    starts = np.random.default_rng(20261112).integers(0, len(cal), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(cal)).reshape(10000, -1)[:, :len(cal)]
    contrasts = {}
    for key in records:
        contrasts[key] = {}
        for label in ('nh4', 'do'):
            difference = errors[key][label] - errors['nh4_global_do_global'][label]
            contrasts[key][label] = {'mean': float(difference.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    result = {**manifest, 'conditions': records, 'brier_minus_global_global': contrasts,
              'both_e61_calibration_identities_passed': True, 'complete': True}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('DECOMPOSITION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
