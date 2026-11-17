"""Fit all preregistered next-DO models and retain every admission failure."""
import argparse
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from forecast_data import ROOT, read_blocks, sha
from do_forecast import fit_model, mean_prediction, compiled_mean, margin, diagnostics, held_report
from rate_io import load_base
from policy_io import verify_sources
from quantile_env import frozen_digest


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    results = workspace / 'results'
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    cert_path = ROOT / 'Lee/experiments/e64/baseline_authentication.json'
    cert = json.loads(cert_path.read_text())
    assert cert['passed'] and cert['policy_seeds'] == list(range(5))
    assert sha(data) == cert['data_sha256'] and frozen_digest(freeze) == cert['freeze_canonical_sha256']
    control_path = Path(__file__).with_name('forecast_check.json')
    control = json.loads(control_path.read_text())
    assert control['passed'] and control['new_environment_queries'] == 0
    verify_sources(control['source_sha256'], ROOT)
    shared = [ROOT / name for name in ('Lee/experiments/e56/rate_model.py',
        'Lee/experiments/e56/targets.py', 'Lee/experiments/e55/fast_cost.py')]
    paths = [*Path(__file__).parent.glob('*.py'), *shared]
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    runtime = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'sklearn': sklearn.__version__, 'torch': torch.__version__, 'device': 'cpu', 'numerical_threads': 1}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    args.out.mkdir(parents=True)
    write(args.out / 'study_manifest.json', {'policy_seeds': list(range(5)), 'source_sha256': sources,
        'code_commit': commit, 'runtime': runtime, 'data_sha256': sha(data),
        'freeze_canonical_sha256': frozen_digest(freeze), 'baseline_certificate_sha256': sha(cert_path),
        'control_certificate_sha256': sha(control_path), 'test_evaluated': False,
        'new_environment_queries': 0, 'controller_evaluation': False})
    records = []
    for seed in range(5):
        assert all(sha(ROOT / name) == expected for name, expected in sources.items())
        write(args.out / 'study_status.json', {'phase': 'fitting', 'completed_policy_seeds': [r['policy_seed'] for r in records],
            'current_policy_seed': seed, 'terminal': False})
        started = time.monotonic()
        run = results / f'e55_study/policies/new_ppolag_fcsrl/seed_{seed}'
        reference = results / f'e42_baselines/new_ppolag_fcsrl/seed_{seed}'
        base, config, _, hashes = load_base(run, freeze, reference)
        assert config['seed'] == seed and hashes == cert['records'][seed]['base_checkpoint_hashes']
        del base
        blocks, provenance = read_blocks(run, data)
        fit, calibration, held = [blocks[name] for name in ('fit', 'calibration', 'held')]
        fit_clock = time.process_time()
        model = fit_model(fit['x'], fit['y'])
        fit_cpu_seconds = time.process_time() - fit_clock
        cal_point = mean_prediction(model, calibration['x'])
        adjustment = margin(calibration['y'], cal_point)
        probe = fit['x'][np.linspace(0, len(fit['x']) - 1, 10000, dtype=int)]
        raw_error = 0.
        for x in (probe, calibration['x'], held['x']):
            expected = mean_prediction(model, x)
            actual = compiled_mean(model, x)
            error = float(np.max(np.abs(actual - expected)))
            raw_error = max(raw_error, error)
            assert error <= 1e-12
        lower, upper = probe.copy(), probe.copy()
        lower[:, -1], upper[:, -1] = -1, 1
        order_violation = float(np.maximum(compiled_mean(model, lower) - compiled_mean(model, upper), 0).max())
        assert order_violation <= 1e-12
        report = held_report(fit, calibration, held, model, adjustment)
        model_held = report['forecasts']['model']
        persistence = report['forecasts']['current_state_persistence']
        checks = {'provenance_and_partition_passed': True, 'compiled_raw_prediction_passed': raw_error <= 1e-12,
            'monotone_action_passed': order_violation <= 1e-12,
            'mse_interval_below_persistence': report['mse_contrasts']['current_state_persistence']['paired_training_date95_ci'][1] < 0,
            'mse_interval_below_fitting_constant': report['mse_contrasts']['fitting_mean_constant']['paired_training_date95_ci'][1] < 0,
            'held_upper_coverage_at_least_95_percent': model_held['upper_coverage'] >= .95,
            'next_do_false_negatives_not_above_persistence': model_held['false_negatives'] <= persistence['false_negatives']}
        folder = args.out / f'seed_{seed}'
        folder.mkdir()
        with (folder / 'do_model.pkl').open('wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
        record = {'policy_seed': seed, 'computation_passed': True, 'admission': {'passed': all(checks.values()), 'checks': checks},
            'model_sha256': sha(folder / 'do_model.pkl'), 'upper_margin': adjustment,
            'calibration': diagnostics(calibration['y'], cal_point, cal_point + adjustment, calibration['x'][:, 2:5].max(-1)),
            'held': report, 'max_original_compiled_mean_difference': raw_error,
            'max_action_order_violation': order_violation, 'base_checkpoint_hashes': hashes,
            'capture_provenance': provenance, 'fit_rows': len(fit['y']), 'calibration_rows': len(calibration['y']),
            'held_rows': len(held['y']), 'base_training_steps': 401280, 'extra_environment_training_steps': 0,
            'new_environment_queries': 0, 'added_offline_supervised_fits': 1, 'boosting_iterations': 200,
            'model_fit_random_state': 7500, 'model_fit_cpu_seconds': fit_cpu_seconds,
            'total_elapsed_seconds': time.monotonic() - started, 'runtime': runtime,
            'test_evaluated': False, 'controller_evaluation': False, 'source_sha256': sources,
            'scope': 'Next-DO forecast on logged simulator actions, not repaired DARROW dynamics or policy quality.'}
        write(folder / 'fit_summary.json', record)
        records.append(record)
        print('NEXT_DO_MODEL_COMPLETE', seed, 'admitted', record['admission']['passed'], checks, flush=True)
    assert all(sha(ROOT / name) == expected for name, expected in sources.items())
    summary = {'completed': True, 'computation_passed': True, 'policy_seeds': list(range(5)),
        'all_models_admitted': all(row['admission']['passed'] for row in records), 'records': records,
        'new_environment_queries': 0, 'test_evaluated': False, 'controller_evaluation': False,
        'baseline_certificate_sha256': sha(cert_path), 'source_sha256': sources, 'code_commit': commit,
        'scope': 'All original seeds retained; a failed admission blocks the conditional controller stage.'}
    write(args.out / 'summary.json', summary)
    write(args.out / 'study_status.json', {'phase': 'complete', 'completed_policy_seeds': list(range(5)),
        'current_policy_seed': None, 'terminal': True})
    print('NEXT_DO_FORECAST_STUDY_COMPLETE', summary['all_models_admitted'], flush=True)


if __name__ == '__main__':
    main()
