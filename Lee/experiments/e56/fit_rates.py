"""Fit the two preregistered regression horizons on identical captured data."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from rate_model import fit_models, FastRate, slow_rates, ROOT
from streams import read_stream
from rate_io import load_base
from quantile_env import frozen_digest


def rate_metrics(y, prediction):
    error = prediction - y
    present = y > 0
    return {'n': len(y), 'nonzero_targets': int(present.sum()),
            'target_mean': float(y.mean()), 'prediction_mean': float(prediction.mean()),
            'mean_bias': float(error.mean()), 'mse': float(np.square(error).mean()),
            'mae': float(np.abs(error).mean()),
            'any_event_ranking_ap': float(average_precision_score(present, prediction)) if present.any() else None,
            'any_event_ranking_auc': float(roc_auc_score(present, prediction)) if len(np.unique(present)) == 2 else None}


def verify(models, probes):
    fast = FastRate(models)
    expected = np.column_stack([model.predict(probes) for model in models])
    np.testing.assert_allclose(fast.raw(probes), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fast.predict(probes), np.clip(expected, 0, 1), atol=1e-12, rtol=1e-12)
    worst = 0.
    for column, sign in {0: 1, 2: -1, 3: -1, 4: -1, 10: -1, 11: -1}.items():
        lower, upper = probes.copy(), probes.copy()
        if column == 11:
            lower[:, column], upper[:, column] = -1, 1
        else:
            lower[:, column] = 0
            upper[:, column] = np.maximum(probes[:, column], 1) * 1.5
        difference = (fast.predict(upper) - fast.predict(lower)) * [sign, -sign]
        worst = max(worst, float(np.maximum(-difference, 0).max()))
    assert worst <= 1e-12
    return {'passed': True, 'probe_rows': len(probes),
            'max_raw_sklearn_difference': float(np.max(np.abs(fast.raw(probes) - expected))),
            'max_clipped_sklearn_difference': float(np.max(np.abs(fast.predict(probes) - np.clip(expected, 0, 1)))),
            'max_coordinate_order_violation': worst}


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'baseline', 'freeze', 'data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--horizon', type=int, choices=(1, 4), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    check = json.loads(Path(__file__).with_name('model_check.json').read_text())
    assert check['passed']
    for name, digest in check['source_sha256'].items():
        assert frozen_digest(Path(__file__).with_name(name)) == digest
    assert frozen_digest(ROOT / 'Lee/experiments/e55/fast_cost.py') == check['shared_inference_source_sha256']
    torch.set_num_threads(1)
    _, config, _, base_hashes = load_base(args.run, args.freeze, args.baseline)
    start = time.monotonic()
    fit, held, provenance = read_stream(args.run, args.data, args.horizon, args.smoke)
    models = fit_models(fit['x'], fit['target'])
    probes = held['x'][np.linspace(0, len(held['x']) - 1, min(10000, len(held['x'])), dtype=int)]
    checks = verify(models, probes)
    prediction = slow_rates(models, held['x'])
    reports = {}
    for index, label in enumerate(('nh4', 'do')):
        current = held['x'][:, 0] > 4 if index == 0 else held['x'][:, 2:5].max(-1) > 2
        y = held['target'][:, index]
        reports[label] = {'held': rate_metrics(y, prediction[:, index]),
                          'current_state_persistence': rate_metrics(y, current.astype(float)),
                          'fit_mean_constant': rate_metrics(y, np.full(len(y), fit['target'][:, index].mean())),
                          'fit_target_mean': float(fit['target'][:, index].mean()),
                          'constant_model': hasattr(models[index], 'value')}
    restored = pickle.loads(pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(slow_rates(models, probes), slow_rates(restored, probes))
    args.out.mkdir(parents=True)
    with (args.out / 'risk_models.pkl').open('wb') as file:
        pickle.dump(models, file, protocol=pickle.HIGHEST_PROTOCOL)
    shared = [*(ROOT / 'Lee/experiments/e55').glob('*.py'),
              *(ROOT / 'Lee/experiments/e44').glob('*.py')]
    summary = {**provenance, 'fit_transitions': len(fit['x']), 'held_transitions': len(held['x']),
               'implementation_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                                                  'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'checks': checks, 'label_diagnostics': reports, 'base_checkpoint_hashes': base_hashes,
               'model_sha256': hashlib.sha256((args.out / 'risk_models.pkl').read_bytes()).hexdigest(),
               'source_sha256': {p.name: frozen_digest(p) for p in Path(__file__).parent.glob('*.py')},
               'shared_source_sha256': {str(p.relative_to(ROOT)): frozen_digest(p) for p in shared},
               'elapsed_seconds': time.monotonic() - start,
               'loss': 'squared_error', 'outputs_clipped': [0, 1], 'no3_retained': True,
               'target_rule': 'Mean of up to horizon transitions including current, stopped at terminal day boundary.',
               'metric_scope': 'Rate MSE. Any-event AP/AUC use rate only as a ranking score, not a calibrated probability of any event.',
               'smoke_rule': 'Fit update1 and diagnose update2 of the full captured base policy; not a full regressor trial.' if args.smoke else None}
    (args.out / 'fit_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('RATE_FIT_PASS', config['seed'], args.horizon, json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
