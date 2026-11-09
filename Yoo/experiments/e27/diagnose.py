"""고정 예측의 구분 능력, 시간 전이와 과거 잔차 보정 효과를 검사한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e12'))
from trial import arrays, load_model, replay, routing_frame, score

MODES = ('start_multi', 'short_mse', 'day_mse')
METHODS = ('none', 'global', 'segment', 'ewma_segment')
SEGMENTS = ((0, 36), (36, 144), (144, 275))
FOLDS = ((22, 23, 44), (44, 45, 66))
HALF_LIFE = 7
ALPHA = 1 - 2 ** (-1 / HALF_LIFE)
ROOT = Path('/home/jio/fcsrl_src/results')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binary(pred, truth, threshold=4.):
    y, p = truth > 4, np.asarray(pred, dtype=float) > threshold
    tp, fp, fn, tn = [int(a.sum()) for a in (p & y, p & ~y, ~p & y, ~p & ~y)]
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp / (tp + fn) if tp + fn else None,
                fpr=fp / (fp + tn) if fp + tn else None)


def discrimination(pred, truth):
    s, y = pred.astype(float).ravel(), (truth > 4).ravel()
    if not y.any() or y.all():
        return dict(auc=None, oracle_at_recall={})
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    result = {}
    for target in (.5, .8):
        i = np.flatnonzero(tpr >= target)[0]
        result[str(target)] = dict(recall=float(tpr[i]), fpr=float(fpr[i]),
                                   threshold=float(np.nextafter(thresholds[i], -np.inf)))
    return dict(auc=float(roc_auc_score(y, s)), oracle_at_recall=result)


def operating_points(pred, truth):
    s, y = pred.astype(float).ravel(), (truth > 4).ravel()
    assert y.any() and not y.all()
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    indices = {f'recall{int(100*r)}': np.flatnonzero(tpr >= r)[0] for r in (.5, .8)}
    candidates = np.flatnonzero(fpr <= .1)
    best = tpr[candidates].max()
    indices['fpr10'] = candidates[tpr[candidates] == best][0]
    return {name: float(np.nextafter(thresholds[i], -np.inf)) for name, i in indices.items()}


def metrics(pred, truth):
    n, t = pred[..., 0], truth[..., 0]
    costs = (pred > np.array([4., 2.])).sum((1, 2))
    actual = (truth > np.array([4., 2.])).sum((1, 2))
    return {**binary(n, t), **discrimination(n, t),
            'nh4_bias': float((n-t).mean()), 'nh4_mae': float(np.abs(n-t).mean()),
            'cost_mae': float(np.abs(costs-actual).mean()),
            'predicted_cost': float(costs.mean()), 'actual_cost': float(actual.mean())}


def segment_bias(pred, truth):
    return np.stack([(pred[:, lo:hi, 0]-truth[:, lo:hi, 0]).mean(1)
                     for lo, hi in SEGMENTS], 1)


def correct(cal_pred, cal_truth, pred, truth, method):
    """그날 결과는 그날 예측 저장 뒤에만 다음 날 보정 상태에 반영한다."""
    out = pred.astype(float).copy()
    errors = segment_bias(cal_pred, cal_truth)
    if method == 'none':
        return out, np.zeros((len(pred), 3))
    if method == 'global':
        state = np.repeat((cal_pred[..., 0]-cal_truth[..., 0]).mean(), 3)
    elif method == 'segment':
        state = errors.mean(0)
    else:
        assert method == 'ewma_segment'
        state = errors[0].copy()
        for error in errors[1:]:
            state = (1-ALPHA)*state + ALPHA*error
    corrections = []
    for day in range(len(pred)):
        corrections.append(state.copy())
        for k, (lo, hi) in enumerate(SEGMENTS):
            out[day, lo:hi, 0] = np.maximum(0, pred[day, lo:hi, 0]-state[k])
        if method == 'ewma_segment':
            # 보정 이전의 고정 모델 잔차를 관측한 뒤 다음 날부터 갱신한다.
            error = segment_bias(pred[day:day+1], truth[day:day+1])[0]
            state = (1-ALPHA)*state + ALPHA*error
    return out, np.asarray(corrections)


def condition_rows(pred, truth, x, bounds):
    p, y = pred[..., 0], truth[..., 0]
    phase, temp = x['phase'][:, 12:-1], x['temp'][:, 12:-1]
    masks = {f'phase{k}': phase == k for k in (0, 1, 2)}
    for k in range(3):
        masks[f'temperature{k}'] = np.digitize(temp, bounds) == k
    for k, (lo, hi) in enumerate(SEGMENTS):
        mask = np.zeros_like(y, dtype=bool)
        mask[:, lo:hi] = True
        masks[f'horizon{k}'] = mask
    rows = []
    for group, mask in masks.items():
        pp, yy = p[mask], y[mask]
        if not len(pp):
            continue
        neg, pos = yy <= 4, yy > 4
        rows.append(dict(group=group, count=len(pp), **binary(pp, yy),
                         bias=float((pp-yy).mean()),
                         negative_bias=float((pp[neg]-yy[neg]).mean()) if neg.any() else None,
                         positive_bias=float((pp[pos]-yy[pos]).mean()) if pos.any() else None,
                         **discrimination(pp, yy)))
    return rows


def paired_bootstrap(predictions, truth):
    """동일 날짜를 두 seed와 두 방법에 함께 재표집한다. 독립 seed 증거가 아니다."""
    rng = np.random.default_rng(27000)
    indices = rng.integers(0, len(truth), (2000, len(truth)))
    y = truth[..., 0] > 4
    positives, negatives = y.sum(1), (~y).sum(1)
    actual = (truth > [4, 2]).sum((1, 2))
    samples = {}
    for method, preds in predictions.items():
        tp = np.stack([((p[..., 0] > 4) & y).sum(1) for p in preds])
        fp = np.stack([((p[..., 0] > 4) & ~y).sum(1) for p in preds])
        error = np.stack([np.abs((p > [4, 2]).sum((1, 2))-actual) for p in preds])
        samples[method] = dict(
            recall=(tp[:, indices].sum(-1)/positives[indices].sum(-1)).mean(0),
            fpr=(fp[:, indices].sum(-1)/negatives[indices].sum(-1)).mean(0),
            cost_mae=error[:, indices].mean(-1).mean(0))
    return {method: {key: dict(mean=float((values-samples['none'][key]).mean()),
                               ci95=np.quantile(values-samples['none'][key], [.025, .975]).tolist())
                     for key, values in sample.items()}
            for method, sample in samples.items() if method != 'none'}


def verify():
    rng = np.random.default_rng(27)
    cp, ct = rng.uniform(0, 8, (10, 275, 2)), rng.uniform(0, 8, (10, 275, 2))
    p, t = rng.uniform(0, 8, (5, 275, 2)), rng.uniform(0, 8, (5, 275, 2))
    for method in METHODS:
        base, _ = correct(cp, ct, p, t, method)
        np.testing.assert_array_equal(base[..., 1], p[..., 1])
        for day in range(5):
            changed = t.copy()
            changed[day:] += 100
            other, _ = correct(cp, ct, p, changed, method)
            np.testing.assert_array_equal(base[:day+1], other[:day+1])
    tp = np.array([1., 2., 3., 4., 4., 6.], dtype=np.float32)
    tt = np.array([1., 2., 5., 5., 1., 5.])
    points = operating_points(tp, tt)
    assert binary(tp, tt, points['recall50'])['recall'] >= .5
    assert binary(tp, tt, points['recall80'])['recall'] >= .8
    assert binary(tp, tt, points['fpr10'])['fpr'] <= .1
    positive_shift = cp.copy()
    positive_shift[..., 0] = ct[..., 0]+2
    corrected, _ = correct(positive_shift, ct, p, t, 'global')
    np.testing.assert_allclose(corrected[..., 0], np.maximum(0, p[..., 0]-2))
    print('PASS causal update, correction sign, DO preservation, threshold ties', flush=True)


def main(out):
    verify()
    if out.exists():
        raise ValueError('Refusing overwrite')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data_path = HERE.parents[1] / 'data/avedore/aved_5min.pkl'
    raw_path = data_path.with_name('aved_raw.csv')
    data = pd.read_pickle(data_path)
    e14 = json.loads((ROOT/'e14/trial/results.json').read_text())
    e26 = json.loads((HERE.parent/'e26/evaluation.json').read_text())
    for path in (data_path, raw_path):
        assert sha(path) == e26['input_sha256'][str(path)] == e14['input_sha256'][str(path)]
    days = dict(inner=e14['partition']['inner_holdout'], val=data['days']['val'])
    assert len(days['inner']) == 66 and len(days['val']) == 43
    assert set(days['inner']).issubset(data['days']['train'])
    assert not (set(days['inner']+days['val']) & set(data['days']['test']))
    xs = {split: arrays(dict(data=data['data'], days=days), split) for split in days}
    truths = {split: np.stack((x['nh4'][:, 13:], x['do'][:, 13:]), -1) for split, x in xs.items()}
    frame = routing_frame(raw_path, end='2024-02-29')
    aux = np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in days['val']])
    assert np.isfinite(aux).all()
    sources = [Path(__file__), *[HERE.parent/f'e{n}/{name}' for n, name in
               ((12, 'trial.py'), (11, 'routing_trial.py'), (10, 'history_trial.py'))]]
    inputs = [data_path, raw_path, ROOT/'e14/trial/results.json', HERE.parent/'e26/evaluation.json']
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  config=dict(modes=MODES, methods=METHODS, segments=SEGMENTS, folds=FOLDS,
                              ewma_half_life_observed_days=HALF_LIFE, seeds=[0, 1]),
                  days=days, test_used=False, model_training=False, calibration_fitted=True,
                  internal_holdout_reused=True, outer_val_reused=True,
                  source_sha256={str(p): sha(p) for p in sources}, input_sha256={str(p): sha(p) for p in inputs},
                  model_sha256={}, cache_sha256={}, raw=[], calibration=[], corrected=[], conditions=[],
                  paired_day_bootstrap={}, checks={})
    all_predictions = {mode: {split: [] for split in days} for mode in MODES}
    for mode in MODES:
        for seed in (0, 1):
            path = (ROOT/'e12/trial'/f'multi_mse_s{seed}.pt' if mode == 'start_multi'
                    else ROOT/'e14/trial'/f'{mode}_s{seed}.pt')
            assert sha(path) == e26['input_sha256'][str(path)]
            report['model_sha256'][str(path)] = sha(path)
            cache = ROOT/'e14/trial'/f'{mode}_s{seed}_holdout.npz'
            report['cache_sha256'][str(cache)] = sha(cache)
            preds = dict(inner=np.load(cache)['pred'],
                         val=replay(load_model(path), xs['val'], aux, 'with_routing', 275))
            for split, pred in preds.items():
                prior = e14 if split == 'inner' else e26
                old = next(r for r in prior['evaluation'] if r['mode'] == mode and r['seed'] == seed
                           and r['split'] == ('inner_holdout' if split == 'inner' else 'val')
                           and r['reset_steps'] == 275)
                now = score(pred, xs[split])
                assert now == {key: old[key] for key in now}
                all_predictions[mode][split].append(pred)
                report['raw'].append(dict(mode=mode, seed=seed, split=split, **metrics(pred, truths[split])))
                for row in condition_rows(pred, truths[split], xs[split], e14['temperature_bounds']):
                    report['conditions'].append(dict(mode=mode, seed=seed, split=split, **row))
            changed = {k: v.copy() for k, v in xs['val'].items()}
            changed['nh4'][:, 13:] += 1000
            changed['do'][:, 13:] += 1000
            np.testing.assert_array_equal(preds['val'], replay(load_model(path), changed, aux, 'with_routing', 275))
            np.savez_compressed(out/f'{mode}_s{seed}.npz', inner=preds['inner'], val=preds['val'])
            print('verified predictions', mode, seed, flush=True)
    for mode in MODES:
        for label in ('inner_oof', 'val'):
            corrected_by_method = {method: [] for method in METHODS}
            for seed in (0, 1):
                inner = all_predictions[mode]['inner'][seed]
                outer = all_predictions[mode]['val'][seed]
                blocks = [(inner[:end], truths['inner'][:end], inner[start:stop], truths['inner'][start:stop],
                           days['inner'][:end], days['inner'][start:stop]) for end, start, stop in FOLDS] if label == 'inner_oof' else [
                           (inner, truths['inner'], outer, truths['val'], days['inner'], days['val'])]
                combined_truth = np.concatenate([b[3] for b in blocks])
                for fold, (cp, ct, pred, truth, cal_days, eval_days) in enumerate(blocks):
                    assert max(cal_days) < min(eval_days)
                    points = operating_points(cp[..., 0], ct[..., 0])
                    for name, threshold in points.items():
                        report['calibration'].append(dict(mode=mode, seed=seed, split=label, fold=fold,
                            operating_point=name, threshold=threshold, cal_days=cal_days, eval_days=eval_days,
                            calibration=binary(cp[..., 0], ct[..., 0], threshold),
                            evaluation=binary(pred[..., 0], truth[..., 0], threshold)))
                for method in METHODS:
                    pieces, correction = [], []
                    for cp, ct, pred, truth, _, _ in blocks:
                        result, offsets = correct(cp, ct, pred, truth, method)
                        pieces.append(result)
                        correction.append(offsets)
                        if method == 'ewma_segment':
                            for day in (0, len(pred)//2, len(pred)-1):
                                changed = truth.copy()
                                changed[day:] += 1000
                                np.testing.assert_array_equal(result[:day+1], correct(cp, ct, pred, changed, method)[0][:day+1])
                    output = np.concatenate(pieces)
                    corrected_by_method[method].append(output)
                    report['corrected'].append(dict(mode=mode, seed=seed, split=label, method=method,
                        **metrics(output, combined_truth), correction_by_day=np.concatenate(correction).tolist()))
            report['paired_day_bootstrap'][f'{mode}/{label}'] = paired_bootstrap(corrected_by_method, combined_truth)
    fingerprints = {**report['input_sha256'], **report['model_sha256'], **report['cache_sha256'], **report['source_sha256']}
    assert all(sha(path) == expected for path, expected in fingerprints.items())
    report['checks'] = dict(original_scores_exact=True, input_source_model_cache_unchanged=True,
                            future_truth_independent=True, calibration_before_evaluation=True,
                            current_day_correction_future_independent=True, synthetic_tests=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    write(out/'diagnosis.json', report)
    print('completed', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        if args.out is None:
            parser.error('--out required')
        main(args.out)
