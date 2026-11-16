"""Fixed calibration and guarded global/context validation for e61."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from context_paths import initial_context, fit_scale, donor_pools, paths
from runtime import ROOT, q, load_model, rollout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e58', 'preflight', 'reference-summary', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    preflight = json.loads(args.preflight.read_text())
    assert preflight['both_selected_banks_reconstructed_bitwise']
    assert preflight['full_pool_original_paths_bitwise_both_banks']
    assert preflight['future_state_exogenous_air_invariance']
    assert preflight['data_sha256'] == digest(args.data)
    for name, expected in preflight['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    summaries, models, banks = {}, {}, {}
    directories = {'e51': args.e51, 'e58': args.e58}
    for name, directory in directories.items():
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['data_sha256'] == preflight['data_sha256']
        assert summary['selected'] == 'pit_b95_a0.25' and summary['test_evaluated'] is False
        for filename, expected in summary['model_hashes'].items():
            assert digest(directory / filename) == expected
        bank = np.load(directory / 'pit_bank.npz')['pit']
        assert hashlib.sha256(bank.tobytes()).hexdigest() == preflight['bank_array_sha256'][name]
        summaries[name], models[name], banks[name] = summary, load_model(directory, name), bank
    raw = pd.read_pickle(args.data)
    reference = json.loads(args.reference_summary.read_text())
    fit, cal = reference['fit_days'], reference['calibration_days']
    for summary in summaries.values():
        assert summary['fit_days'] == fit and summary['calibration_days'] == cal
    assert len(fit) == 158 and len(cal) == 30 and len(raw['days']['val']) == 40
    assert set(fit).isdisjoint(cal)
    assert set(fit + cal).isdisjoint(raw['days']['val'] + raw['days']['test'])
    fitting_context = initial_context(q.arrays(raw['data'], fit))
    mean, sd_context, floor = fit_scale(fitting_context)
    np.testing.assert_array_equal(mean, preflight['fit_context_mean'])
    np.testing.assert_array_equal(sd_context, preflight['fit_context_sd'])
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in fit]).std(0)
    cal_batch = q.arrays(raw['data'], cal)
    donors = initial_context(cal_batch)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                   'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'preflight_sha256': digest(args.preflight), 'data_sha256': digest(args.data),
                'parent_summary_sha256': {name: digest(directory / 'summary.json') for name, directory in directories.items()},
                'fit_days': fit, 'calibration_days': cal, 'test_evaluated': False,
                'fit_context_mean': mean.tolist(), 'fit_context_sd': sd_context.tolist(),
                'sd_floor_used': floor.tolist(), 'primary': 'e51_context5',
                'weight_fitting': False, 'candidate_selection': False}
    write(args.out / 'manifest.json', manifest)

    def evaluate(parent, mode, batch, days, exclude_self, order=False):
        pools = donor_pools(initial_context(batch), donors, days, cal, mean, sd_context,
                            k=5 if mode == 'context5' else None, exclude_self=exclude_self)
        noise = paths(banks[parent], days, pools)
        if mode == 'global' and not exclude_self:
            np.testing.assert_array_equal(noise, q.paths(banks[parent], days, 95))
        trajectory = rollout(models[parent], batch, noise)
        score = q.summarize(trajectory, batch[0], sd)
        score['event_diagnostics'] = q.event_diagnostics(trajectory, batch[0])
        if order:
            low = rollout(models[parent], batch, noise, air_scale=.75)
            high = rollout(models[parent], batch, noise, air_scale=1.25)
            score['recursive_order'] = {
                'nh4_max_order_violation': float(np.maximum(high[..., 0] - low[..., 0], 0).max()),
                'do_max_order_violation': float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())}
        return score

    def calibration_job(task):
        parent, mode = task
        score = evaluate(parent, mode, cal_batch, cal, True)
        key = parent + '_' + mode
        write(args.out / ('calibration_' + key + '.json'), score)
        print('CALIBRATION', key, flush=True)
        return key, score

    with ThreadPoolExecutor(max_workers=2) as pool:
        calibration = dict(pool.map(calibration_job, [(p, mode) for p in ('e51', 'e58') for mode in ('global', 'context5')]))
    days = raw['days']['val']
    batch = q.arrays(raw['data'], days)
    global_scores = {}
    for parent in ('e51', 'e58'):
        score = evaluate(parent, 'global', batch, days, False)
        original = summaries[parent]['validation'][summaries[parent]['selected']]
        for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
            np.testing.assert_allclose(score[key], original[key], rtol=1e-12, atol=1e-14)
        for key in ('nh4_violation', 'do_violation'):
            np.testing.assert_allclose(score[key]['member_pred'], original[key]['member_pred'], rtol=1e-12)
        score['recursive_order'] = original['recursive_order']
        global_scores[parent] = score
        write(args.out / ('validation_global_' + parent + '.json'), score)
        print('GLOBAL_REFERENCE_EXACT', parent, flush=True)
    write(args.out / 'global_reference_check.json', {'both_global_validation_reconstructions_exact': True,
                                                     'candidate_scoring_started': False})
    context_scores = {}
    for parent in ('e51', 'e58'):
        score = evaluate(parent, 'context5', batch, days, False, order=True)
        context_scores[parent] = score
        write(args.out / ('validation_context5_' + parent + '.json'), score)
        print('CONTEXT_VALIDATION', parent, flush=True)
    pools = donor_pools(initial_context(batch), donors, days, cal, mean, sd_context, k=5, exclude_self=False)
    sensitivity = q.summarize(rollout(models['e51'], batch, paths(banks['e51'], days, pools, draws=16), draws=16), batch[0], sd)
    result = {**manifest, 'calibration_leave_one_day_out': calibration,
              'validation_global': global_scores, 'validation_context5': context_scores,
              'primary_sixteen_draw_sensitivity': sensitivity,
              'primary_point_gate': q.gate(context_scores['e51'], reference['validation']['old']),
              'secondary_point_gate': q.gate(context_scores['e58'], reference['validation']['old']),
              'both_global_reference_reconstructions_exact': True,
              'adoption_status': 'pending_independent_full_day_and_event_audits'}
    write(args.out / 'summary.json', result)
    print('EVALUATION_COMPLETE', json.dumps(result['primary_point_gate']), flush=True)


if __name__ == '__main__':
    main()
