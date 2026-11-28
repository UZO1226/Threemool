"""Aggregate rare DO forecast failures without fitting or changing gates."""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e76'))
from residual_forecast import read_blocks, sha, FastRate
from policy_io import verify_sources


def counts(y, point, upper, current, persistence, mask):
    event = y > 2
    alarm = upper > 2
    onset = (current <= 2) & event
    n = int(mask.sum())
    residual = y[mask] - point[mask]
    missed = mask & event & ~alarm
    return {'n': n, 'next_do_events': int((mask & event).sum()),
        'onset_events': int((mask & onset).sum()),
        'false_negatives': int(missed.sum()),
        'onset_false_negatives': int((mask & onset & ~alarm).sum()),
        'false_positives': int((mask & ~event & alarm).sum()),
        'persistence_false_negatives': int((mask & event & ~(persistence > 2)).sum()),
        'model_missed_persistence_detected': int((missed & (persistence > 2)).sum()),
        'model_detected_persistence_missed': int((mask & event & alarm & ~(persistence > 2)).sum()),
        'mse': float(np.mean(residual ** 2)) if n else None,
        'upper_coverage': float(np.mean(y[mask] <= upper[mask])) if n else None,
        'residual_quantiles_50_95_975_99': np.quantile(residual, [.5, .95, .975, .99]).tolist() if n else None,
        'largest_missed_residual': float(np.max(y[missed] - point[missed])) if missed.any() else None}


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'study', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    study = args.study.resolve()
    summary = json.loads((study / 'summary.json').read_text())
    audit = json.loads((study / 'audit.json').read_text())
    assert summary['completed'] and summary['computation_passed'] and audit['passed']
    assert summary['policy_seeds'] == list(range(5))
    assert audit['summary_sha256'] == sha(study / 'summary.json')
    verify_sources(summary['source_sha256'], ROOT)
    records = []
    for seed, stored in enumerate(summary['records']):
        assert stored['policy_seed'] == seed
        folder = study / f'seed_{seed}'
        assert sha(folder / 'do_model.pkl') == stored['model_sha256'] == audit['records'][seed]['model_sha256']
        assert sha(folder / 'fit_summary.json') == audit['records'][seed]['fit_summary_sha256']
        run = args.workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}'
        blocks, provenance = read_blocks(run, args.workspace / 'kaggle_data/darrow_15min.pkl')
        assert provenance == stored['capture_provenance']
        with (folder / 'do_model.pkl').open('rb') as handle:
            model = pickle.load(handle)
        fast = FastRate([model])
        reports = {}
        for label in ('calibration', 'held'):
            block = blocks[label]
            x, y = block['x'], block['y']
            current = x[:, 2:5].max(-1)
            point = np.maximum(current + fast.raw(x)[:, 0], 0.)
            upper = point + stored['upper_margin']
            persistence = current + stored['held']['persistence_margin']
            masks = {'all': np.ones(len(y), dtype=bool), 'current_do_0_1': current < 1,
                'current_do_1_1p5': (current >= 1) & (current < 1.5),
                'current_do_1p5_2': (current >= 1.5) & (current < 2),
                'current_do_2_plus': current >= 2}
            assert sum(int(masks[key].sum()) for key in masks if key != 'all') == len(y)
            reports[label] = {key: counts(y, point, upper, current, persistence, mask) for key, mask in masks.items()}
            old = stored['calibration'] if label == 'calibration' else stored['held']['forecasts']['model']
            for key in ('n', 'mse', 'upper_coverage', 'false_negatives', 'false_positives', 'onset_events', 'onset_false_negatives'):
                np.testing.assert_allclose(reports[label]['all'][key], old[key], rtol=0, atol=1e-12)
        records.append({'policy_seed': seed, 'blocks': reports, 'model_sha256': stored['model_sha256']})
        print(json.dumps({'policy_seed': seed, 'reconstruction_passed': True}), flush=True)
    output = {'completed': True, 'reconstruction_passed': True, 'records': records,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
        'e76_summary_sha256': sha(study / 'summary.json'), 'e76_audit_sha256': sha(study / 'audit.json'),
        'test_evaluated': False, 'new_environment_queries': 0, 'new_model_fitting': False,
        'interpretation': 'Exposed simulator training-capture diagnosis. Associations are not causal or independent generalization evidence.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
