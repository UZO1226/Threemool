"""Match PIT estimation to each fixed blended transition without weight fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import FrozenDynamics
import quantile_dynamics as q

ALPHAS = (0, .25, .5, .75)


def prepare(parent, raw, reference):
    cal = reference['calibration_days']
    assert len(reference['fit_days']) == 158 and len(cal) == 30
    assert set(reference['fit_days']).isdisjoint(cal)
    assert set(reference['fit_days'] + cal).isdisjoint(raw['days']['val'] + raw['days']['test'])
    x, delta = q.build_arrays(raw['data'], cal)
    members = np.repeat(np.arange(5), len(x))
    current = np.tile(x[:, :5], (5, 1))
    values = parent.model.quantiles(np.tile(x, (5, 1)), members)
    target = np.tile(x[:, :5] + delta, (5, 1))
    # Safety states remain inside bounds. NO3 is deterministic across quantiles
    # and its existing final clipping must still be included in the kernel.
    safety = [0, 2, 3, 4]
    assert np.all(current[:, safety] >= 0) and np.all(current[:, safety] <= parent.hi[safety])
    np.testing.assert_array_equal(values[:, :, 1], np.broadcast_to(values[:1, :, 1], values[:, :, 1].shape))
    outside = np.flatnonzero(np.any(current > parent.hi, axis=1))
    probe = np.unique(np.r_[np.arange(min(128, len(current))), outside[:32]])
    u = np.random.default_rng(5101).random((len(probe), 5))
    u[0] = 0
    u[1] = 1
    banks, diagnostics = {}, {}
    for alpha in ALPHAS:
        blended = np.clip(alpha * current[None] + (1 - alpha) * values, 0, parent.hi)
        np.testing.assert_allclose(q.interpolate(blended[:, probe], u),
            np.clip(alpha * current[probe] + (1 - alpha) * q.interpolate(values[:, probe], u), 0, parent.hi),
            rtol=1e-12, atol=1e-12)
        bank = q.pit(blended, target, np.random.default_rng(3801)).reshape(5, len(x), 5)
        bank[..., 1] = .5
        assert np.isfinite(bank).all() and bank.min() >= 0 and bank.max() <= 1
        banks[alpha] = bank
        diagnostics[str(alpha)] = {
            'central90_coverage': ((target >= blended[0]) & (target <= blended[-1])).mean(0).tolist(),
            'outside_quantile_range': ((target < blended[0]) | (target > blended[-1])).mean(0).tolist(),
            'pit_zero_mass': (bank == 0).mean((0, 1)).tolist(),
            'pit_one_mass': (bank == 1).mean((0, 1)).tolist()}
    np.testing.assert_array_equal(banks[0], parent.bank)
    np.testing.assert_array_equal(q.paths(banks[.5], cal, 4), q.paths(banks[.5], cal, 4))
    checks = {'alpha_zero_bank_exact': True, 'actual_clipped_kernel_interpolation_identity': True,
              'safety_calibration_current_states_inside_clip_bounds': True,
              'no3_quantiles_deterministic_exact': True,
              'interpolation_probes_include_outside_no3_states': bool(np.any(current[probe, 1] > parent.hi[1])),
              'interpolation_probes_include_both_cdf_endpoints': True,
              'observed_current_above_clip_count_by_state': (x[:, :5] > parent.hi).sum(0).tolist(),
              'finite_bounded_pit': True, 'noise_reproducibility': True,
              'weight_fitting': False, 'test_evaluated': False,
              'calibration_diagnostics_by_alpha': diagnostics}
    return banks, checks


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'parent-freeze'):
        parser.add_argument('--' + name, type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--preflight-out', type=Path)
    args = parser.parse_args()
    destination = args.preflight_out or args.out
    if destination.exists():
        raise FileExistsError(destination)
    parent = FrozenDynamics(args.parent_freeze)
    raw = pd.read_pickle(args.data)
    ref = json.loads(args.reference_summary.read_text())
    banks, verification = prepare(parent, raw, ref)
    if args.preflight_out is not None:
        args.preflight_out.write_text(json.dumps(verification, indent=2, allow_nan=False))
        print(json.dumps(verification), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                     'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source = json.loads((parent.path / 'summary.json').read_text())
    for name, digest in source['model_hashes'].items():
        source_path = parent.path / name
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == digest
        shutil.copy2(source_path, args.out / name)
        assert hashlib.sha256((args.out / name).read_bytes()).hexdigest() == digest
    verification['copied_parent_weight_hashes_exact'] = True
    fit, cal = ref['fit_days'], ref['calibration_days']
    fitting_x, _ = q.build_arrays(raw['data'], fit)
    verification.update(q.checks(parent.model, fitting_x, strict_order=True))
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in fit]).std(0)
    batch = q.arrays(raw['data'], cal)
    inner, settings, controls = {}, {}, []
    for mode, block in [('uniform', 1), ('pit', 1), ('pit', 4), ('pit', 16), ('pit', 95)]:
        for alpha in ALPHAS:
            key = f'{mode}_b{block}_a{alpha}'
            settings[key] = (mode, block, alpha)
            noise = q.paths(banks[alpha], cal, block, uniform=mode == 'uniform')
            inner[key] = q.summarize(q.rollout(parent.model, batch, noise, alpha), batch[0], sd)
            if mode == 'uniform' or alpha == 0:
                for metric in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
                    np.testing.assert_allclose(inner[key][metric], source['inner'][key][metric], rtol=1e-12, atol=1e-14)
                controls.append(key)
            (args.out / 'inner_progress.json').write_text(json.dumps(inner, indent=2, allow_nan=False))
            print('INNER', key, inner[key]['proper_selection_score'], flush=True)
    assert len(controls) == 8
    verification['unchanged_inner_controls_exact'] = controls
    selected = min(inner, key=lambda key: inner[key]['proper_selection_score'])
    selected_mode, selected_block, selected_alpha = settings[selected]
    np.savez_compressed(args.out / 'pit_bank.npz', pit=banks[selected_alpha])
    days = raw['days']['val']
    batch = q.arrays(raw['data'], days)
    validation = {}
    for key in dict.fromkeys(['uniform_b1_a0', selected]):
        mode, block, alpha = settings[key]
        noise = q.paths(banks[alpha], days, block, uniform=mode == 'uniform')
        trajectory = q.rollout(parent.model, batch, noise, alpha)
        score = q.summarize(trajectory, batch[0], sd)
        score['event_diagnostics'] = q.event_diagnostics(trajectory, batch[0])
        low = q.rollout(parent.model, batch, noise, alpha, air_scale=.75)
        high = q.rollout(parent.model, batch, noise, alpha, air_scale=1.25)
        score['recursive_order'] = {
            'nh4_max_order_violation': float(np.maximum(high[..., 0] - low[..., 0], 0).max()),
            'do_max_order_violation': float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())}
        validation[key] = score
    sensitivity = q.summarize(q.rollout(parent.model, batch,
        q.paths(banks[selected_alpha], days, selected_block, draws=16, uniform=selected_mode == 'uniform'),
        selected_alpha, draws=16), batch[0], sd)
    result = {'selected': selected, 'gate': q.gate(validation[selected], ref['validation']['old']),
              'inner': inner, 'validation': validation, 'sixteen_draw_sensitivity': sensitivity,
              'checks': verification, 'test_evaluated': False, 'weight_fitting': False,
              'fit_days': fit, 'calibration_days': cal, 'code_commit': commit,
              'quantile_levels': q.LEVELS.tolist(), 'model_hashes': source['model_hashes'],
              'parent_freeze_sha256': hashlib.sha256(args.parent_freeze.read_bytes()).hexdigest(),
              'calibration_rule': 'PIT estimated after alpha mixing with current observed state, bank per alpha.',
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('RESULT', selected, json.dumps(result['gate']), flush=True)


if __name__ == '__main__':
    main()
