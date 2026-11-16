"""Evaluate a fixed full-strength paired tail-innovation candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from tail_dynamics import preflight, paired_paths, rollout, ALPHAS, q
from quantile_env import FrozenDynamics

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'parent-freeze', 'e51-runs'):
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
    pits, tails, verification = preflight(parent, raw, ref, args.e51_runs)
    if args.preflight_out is not None:
        args.preflight_out.write_text(json.dumps(verification, indent=2, allow_nan=False))
        print(json.dumps(verification), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                     'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    original = json.loads((parent.path / 'summary.json').read_text())
    for name, digest in original['model_hashes'].items():
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
    inner, settings, uniform_controls = {}, {}, []
    for mode, block in [('uniform', 1), ('pit', 1), ('pit', 4), ('pit', 16), ('pit', 95)]:
        for alpha in ALPHAS:
            key = f'{mode}_b{block}_a{alpha}'
            settings[key] = (mode, block, alpha)
            noise = paired_paths(pits[alpha], tails[alpha], cal, block, uniform=mode == 'uniform')
            inner[key] = q.summarize(rollout(parent.model, batch, noise, alpha), batch[0], sd)
            if mode == 'uniform':
                for metric in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
                    np.testing.assert_allclose(inner[key][metric], original['inner'][key][metric], rtol=1e-12, atol=1e-14)
                uniform_controls.append(key)
            (args.out / 'inner_progress.json').write_text(json.dumps(inner, indent=2, allow_nan=False))
            print('INNER', key, inner[key]['proper_selection_score'], flush=True)
    assert len(uniform_controls) == 4
    verification['uniform_controls_exact'] = uniform_controls
    selected = min(inner, key=lambda key: inner[key]['proper_selection_score'])
    selected_mode, selected_block, selected_alpha = settings[selected]
    np.savez_compressed(args.out / 'paired_bank.npz', pit=pits[selected_alpha], tail=tails[selected_alpha])
    days = raw['days']['val']
    batch = q.arrays(raw['data'], days)
    validation = {}
    for key in dict.fromkeys(['uniform_b1_a0', selected]):
        mode, block, alpha = settings[key]
        noise = paired_paths(pits[alpha], tails[alpha], days, block, uniform=mode == 'uniform')
        trajectory, clipping = rollout(parent.model, batch, noise, alpha, diagnostics=True)
        score = q.summarize(trajectory, batch[0], sd)
        score['event_diagnostics'] = q.event_diagnostics(trajectory, batch[0])
        score['physical_clip_diagnostics'] = clipping
        low = rollout(parent.model, batch, noise, alpha, air_scale=.75)
        high = rollout(parent.model, batch, noise, alpha, air_scale=1.25)
        score['recursive_order'] = {
            'nh4_max_order_violation': float(np.maximum(high[..., 0] - low[..., 0], 0).max()),
            'do_max_order_violation': float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())}
        validation[key] = score
    sensitivity = q.summarize(rollout(parent.model, batch,
        paired_paths(pits[selected_alpha], tails[selected_alpha], days, selected_block, draws=16,
                     uniform=selected_mode == 'uniform'), selected_alpha, draws=16), batch[0], sd)
    result = {'selected': selected, 'gate': q.gate(validation[selected], ref['validation']['old']),
              'inner': inner, 'validation': validation, 'sixteen_draw_sensitivity': sensitivity,
              'checks': verification, 'test_evaluated': False, 'weight_fitting': False,
              'tail_variant': 'Fixed full strength, no searched multiplier, NO3 innovation zero.',
              'fit_days': fit, 'calibration_days': cal, 'code_commit': commit,
              'quantile_levels': q.LEVELS.tolist(), 'model_hashes': original['model_hashes'],
              'parent_freeze_sha256': hashlib.sha256(args.parent_freeze.read_bytes()).hexdigest(),
              'e51_reference_summary_sha256': hashlib.sha256((args.e51_runs / 'summary.json').read_bytes()).hexdigest(),
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('RESULT', selected, json.dumps(result['gate']), flush=True)


if __name__ == '__main__':
    main()
