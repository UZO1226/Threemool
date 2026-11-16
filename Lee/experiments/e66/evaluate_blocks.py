"""Evaluate fixed context blocks without selecting parameters on validation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from exogenous_blocks import CONTEXT, donor_pools, paths
from check_blocks import digest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import FrozenDynamics
import quantile_dynamics as q


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'freeze', 'preflight', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    workspace = args.workspace.resolve()
    preflight = json.loads(args.preflight.read_text())
    assert preflight['passed'] and preflight['calibration_only']
    assert preflight['independent_all_calibration_paths_exact']
    for name, expected in preflight['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    data_path = workspace / 'kaggle_data/darrow_15min.pkl'
    assert digest(data_path) == preflight['data_sha256']
    frozen = FrozenDynamics(args.freeze)
    assert digest(args.freeze) == preflight['freeze_canonical_sha256']
    assert hashlib.sha256(frozen.bank.tobytes()).hexdigest() == preflight['bank_array_sha256']
    raw = pd.read_pickle(data_path)
    fit, cal = preflight['fit_days'], preflight['calibration_days']
    mean, sd_context = np.array(preflight['fit_context_mean']), np.array(preflight['fit_context_sd'])
    donors = np.stack([raw['data'].loc[day][CONTEXT].to_numpy(float)[:95] for day in cal])
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in fit]).std(0)
    args.out.mkdir(parents=True)
    manifest = {'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'preflight_sha256': digest(args.preflight), 'data_sha256': digest(data_path),
                'freeze_canonical_sha256': digest(args.freeze), 'test_evaluated': False,
                'primary': 'context32_block4', 'new_model_fitting': False, 'candidate_selection': False,
                'fit_days': fit, 'calibration_days': cal,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(args.out / 'manifest.json', manifest)
    def score(days, mode, exclude_self, draws=8, order=False):
        batch = q.arrays(raw['data'], days)
        context = batch[1][:, :95, :4]
        pool = donor_pools(context, donors, days, cal, mean, sd_context,
                           k=32 if mode == 'context32_block4' else None, exclude_self=exclude_self)
        noise = paths(frozen.bank, days, pool, draws)
        trajectory = q.rollout(frozen.model, batch, noise, .5, draws=draws)
        result = q.summarize(trajectory, batch[0], sd)
        result['event_diagnostics'] = q.event_diagnostics(trajectory, batch[0])
        if order:
            low = q.rollout(frozen.model, batch, noise, .5, draws=draws, air_scale=.75)
            high = q.rollout(frozen.model, batch, noise, .5, draws=draws, air_scale=1.25)
            result['recursive_order'] = {
                'nh4_max_order_violation': float(np.maximum(high[..., 0] - low[..., 0], 0).max()),
                'do_max_order_violation': float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())}
        return result
    calibration = {}
    for mode in ('global_block4', 'context32_block4'):
        calibration[mode] = score(cal, mode, True)
        write(args.out / f'calibration_{mode}.json', calibration[mode])
        print('CALIBRATION_COMPLETE', mode, flush=True)
    days = raw['days']['val']
    batch = q.arrays(raw['data'], days)
    original = q.summarize(q.rollout(frozen.model, batch, q.paths(frozen.bank, days, 95), .5), batch[0], sd)
    parent = json.loads((frozen.path / 'summary.json').read_text())
    reference = parent['validation'][parent['selected']]
    for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
        np.testing.assert_allclose(original[key], reference[key], rtol=1e-12, atol=1e-14)
    for key in ('nh4_violation', 'do_violation'):
        np.testing.assert_allclose(original[key]['member_pred'], reference[key]['member_pred'], rtol=1e-12)
    write(args.out / 'frozen_reference.json', original)
    print('FROZEN_REFERENCE_EXACT', flush=True)
    validation = {}
    for mode in ('global_block4', 'context32_block4'):
        validation[mode] = score(days, mode, False, order=True)
        write(args.out / f'validation_{mode}.json', validation[mode])
        print('VALIDATION_COMPLETE', mode, flush=True)
    sensitivity = score(days, 'context32_block4', False, draws=16)
    reference_summary = json.loads((workspace / 'results/e17_contractive/summary.json').read_text())
    result = {**manifest, 'calibration_leave_one_day_out': calibration,
              'validation': validation, 'frozen_reference': original,
              'frozen_reference_reconstructed_exact': True,
              'primary_sixteen_draw_sensitivity': sensitivity,
              'primary_point_gate': q.gate(validation['context32_block4'], reference_summary['validation']['old']),
              'adoption_status': 'pending_independent_full_day_and_event_audits',
              'scope': 'Exploratory reused validation; no policy or physical plant gain.'}
    write(args.out / 'summary.json', result)
    print('BLOCK_EVALUATION_COMPLETE', json.dumps(result['primary_point_gate']), flush=True)


if __name__ == '__main__':
    main()
