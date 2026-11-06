"""초기 한 시간의 혁신 잔차가 이후 NH4 편향을 예측하는지 먼저 검사한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e14'))
from day_trial import arrays, pack, load_model, routing_frame, sha, write
from trial import replay, score

CFG = dict(seeds=[0, 1], ridge_penalty=.1, minimum_pooled_mse_gain=.05,
           minimum_better_folds=2, folds=[dict(train_end=131, test_start=132, test_end=175),
                                         dict(train_end=175, test_start=176, test_end=219),
                                         dict(train_end=219, test_start=220, test_end=263)],
           primary='context_residual', target='mean_prediction_minus_truth_nh4_steps13_to287')
FEATURES = ['nh4_at_0100', 'do_at_0100', 'initial_temperature', 'initial_flow',
            'initial_phase0_fraction', 'initial_phase2_fraction',
            'initial_nh4_true_minus_prediction', 'initial_do_true_minus_prediction']
MODES = dict(constant=[], context=list(range(6)), residual=[6, 7], context_residual=list(range(8)))


@torch.no_grad()
def initial_features(model, state, exog, extra):
    """측정값은 00:00부터 01:00까지, 전이는 00:00부터 00:55까지만 사용한다."""
    pred = model(state[:, :12].reshape(-1, 2), exog[:, :12].reshape(-1, 6), extra[:, :12].reshape(-1, 4, 2))
    innovation = state[:, 1:13] - pred.reshape(-1, 12, 2)
    phase = exog[:, :13, 1]
    context = torch.stack([state[:, 12, 0], state[:, 12, 1], exog[:, :13, 2].mean(1),
                           exog[:, :13, 3].mean(1), (phase == 0).float().mean(1),
                           (phase == 2).float().mean(1)], 1)
    return torch.cat([context, innovation.mean(1)], 1).numpy(), innovation.numpy()


def fit_ridge(x, y, columns):
    x = np.asarray(x[:, columns], dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = x.mean(0)
    sd = x.std(0)
    sd = np.where(sd > 1e-8, sd, 1.)
    z = (x-mu)/sd
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z / len(z) + CFG['ridge_penalty'] * np.eye(len(columns)),
                                   z.T @ (y-intercept) / len(z)) if columns else np.array([])
    return dict(columns=columns, mean=mu.tolist(), scale=sd.tolist(), intercept=intercept,
                coefficients=coefficients.tolist())


def predict(fit, x):
    z = (x[:, fit['columns']] - np.array(fit['mean'])) / np.array(fit['scale'])
    return fit['intercept'] + z @ np.array(fit['coefficients'])


def assessment(x, y):
    folds, records = [], {mode: [] for mode in MODES}
    for i, f in enumerate(CFG['folds']):
        train, test = np.arange(f['train_end']), np.arange(f['test_start'], f['test_end'])
        assert train.max()+1 < test.min()
        row = dict(fold=i, train_days=len(train), test_days=len(test), mse={}, fits={})
        for mode, columns in MODES.items():
            fit = fit_ridge(x[train], y[train], columns)
            estimate = predict(fit, x[test])
            squared = (estimate-y[test])**2
            row['mse'][mode] = float(squared.mean())
            row['fits'][mode] = fit
            records[mode] += [dict(day_index=int(d), prediction=float(p), target=float(t), squared_error=float(e))
                              for d, p, t, e in zip(test, estimate, y[test], squared)]
        folds.append(row)
    return dict(folds=folds, pooled_mse={mode: float(np.mean([r['squared_error'] for r in rows])) for mode, rows in records.items()},
                oof_predictions=records)


def signal_gate(rows):
    checks = []
    for row in rows:
        for base in ('constant', 'context'):
            gain = 1-row['pooled_mse'][CFG['primary']]/max(row['pooled_mse'][base], 1e-12)
            better = sum(f['mse'][CFG['primary']] < f['mse'][base] for f in row['folds'])
            checks.append(dict(seed=row['seed'], base=base, pooled_gain=float(gain), better_folds=int(better),
                               passed=bool(gain >= CFG['minimum_pooled_mse_gain'] and better >= CFG['minimum_better_folds'])))
    return dict(checks=checks, passed=bool(len(rows) == len(CFG['seeds']) and all(c['passed'] for c in checks)))


def main(a):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if a.out.exists(): raise ValueError(f'Refusing overwrite {a.out}')
    a.out.mkdir(parents=True)
    prior = json.loads((a.e14/'results.json').read_text())
    inputs = {str(p): sha(p) for p in (a.raw, a.data)}
    assert inputs == prior['input_sha256']
    data = pd.read_pickle(a.data)
    partition = prior['partition']
    assert len(partition['fit']) == 263
    x = arrays(dict(data=data['data'], days=partition), 'fit')
    state, exog = pack(x)
    frame = routing_frame(a.raw)
    aux = np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in partition['fit']])
    tensors = tuple(torch.tensor(z).float() for z in (state, exog, aux))
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), config=CFG, features=FEATURES,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=sha(__file__), input_sha256=inputs, partition=partition,
                  inner_holdout_used=False, outer_val_used=False, test_used=False,
                  base_model_seen_all_fit=True, base_model_updated=False,
                  observer_training_performed=False, calibration_regressions_fitted=True,
                  seeds=[], model_sha256={})
    for seed in CFG['seeds']:
        path = a.e14/f'day_mse_s{seed}.pt'
        expected = next(r['model_sha256'] for r in prior['training'] if r['mode']=='day_mse' and r['seed']==seed)
        assert sha(path) == expected
        report['model_sha256'][str(path)] = expected
        model = load_model(path)
        features, innovation = initial_features(model, *tensors)
        perturbed = [v.clone() for v in tensors]
        for v in perturbed: v[:, 13:] += 1000
        np.testing.assert_array_equal(features, initial_features(model, *perturbed)[0])
        pred = replay(model, x, aux, 'with_routing', 275)
        original = next(r for r in prior['evaluation'] if r['mode']=='day_mse' and r['seed']==seed and r['split']=='fit' and r['reset_steps']==275)
        assert all(original[k] == v for k, v in score(pred, x).items())
        error = pred[:, :, 0]-state[:, 13:, 0]
        target = error.mean(1)
        result = assessment(features, target)
        result.update(seed=seed, fit_days=partition['fit'], initial_features=features.tolist(),
                      target_daily_bias=target.tolist(), target_bias_by_segment=np.stack([error[:, :36].mean(1), error[:, 36:144].mean(1), error[:, 144:].mean(1)], 1).tolist())
        report['seeds'].append(result)
        np.savez_compressed(a.out/f'fit_signal_s{seed}.npz', features=features, innovation=innovation, prediction=pred, target=target)
        print(json.dumps(dict(seed=seed, pooled_mse=result['pooled_mse'])), flush=True)
    report['gate'] = signal_gate(report['seeds'])
    report['checks'] = dict(initial_features_future_independent=True, fixed_model_replay_exact=True,
                            chronological_regression_folds=True, input_model_hashes_verified=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    write(a.out/'signal_summary.json', report)
    print(json.dumps(report['gate']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('raw', 'data', 'e14', 'out'): p.add_argument('--'+key, type=Path, required=True)
    main(p.parse_args())
