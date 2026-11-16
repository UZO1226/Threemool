"""Fixed alpha-matched calibration and validation for the conditional DO tail."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from upper_tail import ROOT, UpperDynamics, interpolate, pit, q, LEVELS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dyn, raw, days):
    x, delta = q.build_arrays(raw['data'], days)
    members = np.repeat(np.arange(5), len(x))
    current = np.tile(x[:, :5], (5, 1))
    values = dyn.quantiles(np.tile(x, (5, 1)), members)
    target = np.tile(x[:, :5] + delta, (5, 1))
    safety = [0, 2, 3, 4]
    assert np.all(current[:, safety] <= dyn.hi[safety]) and np.all(current[:, safety] >= 0)
    np.testing.assert_array_equal(values[:, :, 1], np.broadcast_to(values[:1, :, 1], values[:, :, 1].shape))
    outside = np.flatnonzero(np.any(current > dyn.hi, axis=1))
    probe = np.unique(np.r_[np.arange(128), outside[:32]])
    u = np.random.default_rng(5804).random((len(probe), 5))
    u[0] = 0
    u[1] = 1
    banks = {}
    for alpha in (0, .25, .5, .75):
        blended = np.clip(alpha * current[None] + (1 - alpha) * values, 0, dyn.hi)
        np.testing.assert_allclose(interpolate(blended[:, probe], u),
            np.clip(alpha * current[probe] + (1 - alpha) * interpolate(values[:, probe], u), 0, dyn.hi),
            rtol=1e-12, atol=1e-12)
        bank = pit(blended, target, np.random.default_rng(3801)).reshape(5, len(x), 5)
        bank[..., 1] = .5
        assert np.isfinite(bank).all() and bank.min() >= 0 and bank.max() <= 1
        banks[alpha] = bank
    raw_upper = np.clip(current + dyn.upper.delta(np.tile(x, (5, 1)), members), 0, dyn.hi)
    return banks, {'actual_clipped_mixing_interpolation_identity': True,
                   'no3_quantiles_deterministic_exact': True,
                   'outside_no3_probe_included': bool(np.any(current[probe, 1] > dyn.hi[1])),
                   'calibration_upper_head_below_backbone_fraction_do':
                       (raw_upper[:, 2:] < values[-2, :, 2:]).mean(0).tolist()}


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'runs'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.runs
    if (out / 'evaluation_manifest.json').exists() or (out / 'summary.json').exists():
        raise FileExistsError('Evaluation has already started; inspect its actual process before resuming')
    fit_summary = json.loads((out / 'fit_summary.json').read_text())
    assert fit_summary['data_sha256'] == digest(args.data)
    for name, expected in fit_summary['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    for name, expected in fit_summary['model_hashes'].items():
        assert digest(out / name) == expected
    raw = pd.read_pickle(args.data)
    reference = json.loads(args.reference_summary.read_text())
    assert fit_summary['fit_days'] == reference['fit_days']
    assert fit_summary['calibration_days'] == reference['calibration_days']
    manifest = {'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                    'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'fit_summary_sha256': digest(out / 'fit_summary.json'), 'test_evaluated': False}
    (out / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2))
    dyn = UpperDynamics(q.QuantileDynamics(out), q.GBMDynamics(out / 'upper_do_0.99.pkl'))
    banks, checks = prepare(dyn, raw, reference['calibration_days'])
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float)
                         for day in reference['fit_days']]).std(0)
    batch = q.arrays(raw['data'], reference['calibration_days'])
    inner, settings = {}, {}
    with patch.object(q, 'interpolate', interpolate):
        for mode, block in [('uniform', 1), ('pit', 1), ('pit', 4), ('pit', 16), ('pit', 95)]:
            for alpha in (0, .25, .5, .75):
                key = f'{mode}_b{block}_a{alpha}'
                settings[key] = (mode, block, alpha)
                noise = q.paths(banks[alpha], reference['calibration_days'], block, uniform=mode == 'uniform')
                inner[key] = q.summarize(q.rollout(dyn, batch, noise, alpha), batch[0], sd)
                (out / 'inner_progress.json').write_text(json.dumps(inner, indent=2, allow_nan=False))
                print('INNER', key, inner[key]['proper_selection_score'], flush=True)
        selected = min(inner, key=lambda key: inner[key]['proper_selection_score'])
        mode, block, alpha = settings[selected]
        (out / 'selection.json').write_text(json.dumps({'selected': selected, 'selection_split': 'calibration'}))
        np.savez_compressed(out / 'pit_bank.npz', pit=banks[alpha])
        days = raw['days']['val']
        batch = q.arrays(raw['data'], days)
        validation = {}
        for key in dict.fromkeys(['uniform_b1_a0', selected]):
            mode, block, alpha = settings[key]
            noise = q.paths(banks[alpha], days, block, uniform=mode == 'uniform')
            trajectory = q.rollout(dyn, batch, noise, alpha)
            score = q.summarize(trajectory, batch[0], sd)
            score['event_diagnostics'] = q.event_diagnostics(trajectory, batch[0])
            low = q.rollout(dyn, batch, noise, alpha, air_scale=.75)
            high = q.rollout(dyn, batch, noise, alpha, air_scale=1.25)
            score['recursive_order'] = {
                'nh4_max_order_violation': float(np.maximum(high[..., 0] - low[..., 0], 0).max()),
                'do_max_order_violation': float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())}
            validation[key] = score
            print('VALIDATION', key, flush=True)
        mode, block, alpha = settings[selected]
        sensitivity = q.summarize(q.rollout(dyn, batch, q.paths(banks[alpha], days, block,
            draws=16, uniform=mode == 'uniform'), alpha, draws=16), batch[0], sd)
    result = {'selected': selected, 'inner': inner, 'validation': validation,
              'gate': q.gate(validation[selected], reference['validation']['old']),
              'sixteen_draw_sensitivity': sensitivity, 'checks': checks,
              'fit_days': reference['fit_days'], 'calibration_days': reference['calibration_days'],
              'quantile_levels': LEVELS.tolist(), 'model_hashes': fit_summary['model_hashes'],
              'data_sha256': digest(args.data), 'code_commit': manifest['code_commit'],
              'test_evaluated': False, 'new_regressor_count': 15,
              'adoption_status': 'pending_independent_audit_and_preregistered_event_criteria'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('RESULT', selected, json.dumps(result['gate']), flush=True)


if __name__ == '__main__':
    main()
