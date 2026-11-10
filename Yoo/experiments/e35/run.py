"""DO 직접 폭기 구간 제약만 수정해 15개 GBM을 재학습하고 검증 날짜를 재생한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mapping import monotone_constraints, legacy_constraints, FEATURES, DO_AIR, verify, STATE
from common import save, sha
from surrogate_dw import build_arrays, GBMDynamics
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes


def confusion(truth, pred):
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    positives, negatives = int(truth.sum()), int((~truth).sum())
    return {'tp': tp, 'fp': fp, 'positives': positives, 'negatives': negatives,
            'recall_pct': 100 * tp / positives if positives else None,
            'fpr_pct': 100 * fp / negatives if negatives else None,
            'observed_pct': float(truth.mean() * 100), 'predicted_pct': float(pred.mean() * 100)}


def summarize(actual, pred):
    diff = pred - actual
    out = {'mae': dict(zip(STATE, np.abs(diff).mean((0, 1)).tolist())),
           'rmse': dict(zip(STATE, np.sqrt((diff**2).mean((0, 1))).tolist())),
           'confusion': {}, 'episode_mean_counts': {}}
    for name, t, p in [('NH4', actual[:, :, 0] > 4, pred[:, :, 0] > 4),
                       ('DO', actual[:, :, 2:].max(2) > 2, pred[:, :, 2:].max(2) > 2)]:
        out['confusion'][name] = confusion(t, p)
        out['episode_mean_counts'][name] = {'observed': float(t.sum(1).mean()), 'predicted': float(p.sum(1).mean())}
    for j in range(3):
        out['confusion'][f'DO{j+1}'] = confusion(actual[:, :, j+2] > 2, pred[:, :, j+2] > 2)
    return out


def evaluate(prepared, model_path):
    data = pd.read_pickle(prepared / 'data.pkl')
    splits, days = load_days(prepared / 'data.pkl')
    x, delta = build_arrays(data['data'], splits['val'])
    dyn = GBMDynamics(model_path)
    preds = np.stack([dyn.delta(x, np.full(len(x), m)) for m in range(dyn.n_members)])
    mean = preds.mean(0)
    r2 = 1 - ((delta-mean)**2).sum(0) / ((delta-delta.mean(0))**2).sum(0)
    single = np.clip(x[:, :5] + mean, 0, dyn.hi)
    truth = x[:, :5] + delta
    one = summarize(truth[None], single[None])
    one['delta_r2'] = dict(zip(STATE, r2.tolist()))
    env = BatchEnv(days, Dynamics(model_path))
    env.reset(all_episodes(splits['val'], dyn.n_members))
    actual = np.stack([days[d]['state'][1:] for d, _ in env.episodes])
    traj, volume = [], np.zeros(len(env.episodes))
    for _ in range(env.T):
        _, _, _, _, info = env.step(env.historical_action())
        volume += info['air'] * .25
        traj.append(env.state.copy())
    pred = np.stack(traj, 1)
    closed = summarize(actual, pred)
    closed['air_m3_per_23_75h'] = float(volume.mean())
    closed['upper_clip_pct'] = dict(zip(STATE, (100 * np.isclose(pred, dyn.hi).mean((0, 1))).tolist()))
    checks = {}
    for j, target in enumerate(('do1', 'do2', 'do3'), 2):
        ids = np.flatnonzero(monotone_constraints(j))
        low, high = x.copy(), x.copy()
        low[:, ids] *= .75
        high[:, ids] *= 1.25
        rates = []
        for m in range(dyn.n_members):
            member = np.full(len(x), m)
            rates.append(float(100 * (dyn.delta(high, member)[:, j] < dyn.delta(low, member)[:, j] - 1e-10).mean()))
        checks[target] = rates
    return {'one_step_ensemble_mean': one, 'rollout_member_average': closed,
            'physical_zone_wrong_sign_pct_by_member': checks}, actual, pred


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'corrected_surrogate.pkl').exists():
        raise FileExistsError('Use a new output directory to preserve this experiment.')
    verify()
    old_path = a.prepared / 'surrogate.pkl'
    with old_path.open('rb') as f:
        old = pickle.load(f)
    assert len(old['models']) == 5
    data = pd.read_pickle(a.prepared / 'data.pkl')
    x, y = build_arrays(data['data'], data['days']['train'])
    xv, _ = build_arrays(data['data'], data['days']['val'])
    for row in old['models']:
        for j, model in enumerate(row):
            assert np.array_equal(model.monotonic_cst, legacy_constraints(j))
    fixed = pickle.loads(pickle.dumps(old))
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                'input_sha256': {p.name: sha(p) for p in (old_path, a.prepared / 'data.pkl')},
                'mapping': DO_AIR, 'features': FEATURES, 'members': 5,
                'changed_models': 'DO1/DO2/DO3 only; NH4 and NO3 retained exactly',
                'selection': 'fixed e30 hyperparameters, same train-only early stopping',
                'train_days': len(data['days']['train']), 'val_days': len(data['days']['val']),
                'test_evaluated': False, 'policy_training': False}
    save(a.out / 'manifest.json', manifest)
    log = []
    for m in range(5):
        ids = np.random.RandomState(m).choice(len(x), int(.8 * len(x)), replace=False)
        for j in (2, 3, 4):
            params = old['models'][m][j].get_params()
            if m == 0 and j == 2:
                replica = HistGradientBoostingRegressor(**params).fit(x[ids], y[ids, j])
                assert np.array_equal(replica.predict(xv), old['models'][m][j].predict(xv))
                manifest['legacy_refit_prediction_exact'] = True
                save(a.out / 'manifest.json', manifest)
            params['monotonic_cst'] = monotone_constraints(j)
            fixed['models'][m][j] = HistGradientBoostingRegressor(**params).fit(x[ids], y[ids, j])
            log.append({'member': m, 'target': STATE[j], 'iterations': fixed['models'][m][j].n_iter_,
                        'elapsed_seconds': time.monotonic() - start})
            save(a.out / 'progress.json', {'phase': 'surrogate_training', 'completed': len(log), 'total': 15, 'log': log})
            print('FIT', m, STATE[j], f'{time.monotonic()-start:.1f}s', flush=True)
    for m in range(5):
        for j in (0, 1):
            assert pickle.dumps(old['models'][m][j]) == pickle.dumps(fixed['models'][m][j])
    assert np.array_equal(old['hi'], fixed['hi'])
    with (a.out / 'corrected_surrogate.pkl').open('wb') as f:
        pickle.dump(fixed, f)
    save(a.out / 'progress.json', {'phase': 'validation', 'completed': 15, 'total': 15})
    results = {}
    predictions = {}
    for name, path in [('legacy', old_path), ('corrected', a.out / 'corrected_surrogate.pkl')]:
        results[name], actual, pred = evaluate(a.prepared, path)
        predictions[name] = pred
        print('EVAL', name, json.dumps(results[name]['rollout_member_average']), flush=True)
    # The day is the sampling unit; ensemble members and adjacent times are dependent.
    error_difference = np.abs(predictions['corrected'] - actual) - np.abs(predictions['legacy'] - actual)
    per_day = error_difference.reshape(40, 5, 95, 5).mean((1, 2))
    rng = np.random.default_rng(3500)
    boot = per_day[rng.integers(40, size=(2000, 40))].mean(1)
    intervals = {key: {'mean': float(per_day[:, j].mean()),
                      'paired_day_bootstrap_95ci': np.quantile(boot[:, j], [.025, .975]).tolist()}
                 for j, key in enumerate(STATE)}
    assert all(v == 0 for row in results['corrected']['physical_zone_wrong_sign_pct_by_member'].values() for v in row)
    assert sha(old_path) == manifest['input_sha256']['surrogate.pkl']
    output = {'results': results, 'corrected_minus_legacy_rollout_mae': intervals,
              'checks_passed': True, 'legacy_nh4_no3_weights_unchanged': True,
              'corrected_model_sha256': sha(a.out / 'corrected_surrogate.pkl'),
              'no_automatic_adoption': True, 'policy_training': False, 'test_evaluated': False}
    save(a.out / 'comparison.json', output)
    np.savez_compressed(a.out / 'rollouts.npz', actual=actual, **predictions)
    md = ['# e35 DO 직접 폭기 구간 수정 대조', '',
          '같은188일로 DO모델15개만 재학습, 같은검증40일 x5구성원으로 기록 행동을 재생했다.',
          'NH4/NO3 모델 가중치, 데이터, 하이퍼파라미터와 상태 clipping 상한은 고정했다.', '',
          '| 지표 | 기존 대응 | 수정 대응 |', '| --- | --- | --- |']
    for key in STATE:
        v = [results[c]['rollout_member_average']['mae'][key] for c in ('legacy', 'corrected')]
        md.append(f'| {key} 장기 재생 MAE | {v[0]:.5f} | {v[1]:.5f} |')
    for name in ('NH4', 'DO'):
        for metric in ('recall_pct', 'fpr_pct', 'predicted_pct'):
            v = [results[c]['rollout_member_average']['confusion'][name][metric] for c in ('legacy', 'corrected')]
            md.append(f'| {name} {metric} | {v[0]:.3f} | {v[1]:.3f} |')
    md += ['', '변화량 MAE의 날짜별 paired bootstrap95%구간은 comparison.json에 기록했다.',
           '이는 반복 사용한 개발 검증 날짜의 진단이다. 독립 최종 검증이나 현장 안전 증명이 아니다.',
           '분류 기준 NH4>4/DO>2는 과거 실험과 대조하기 위한 값이며 현장 운전 기준으로 새로 확정한 것이 아니다.', '']
    (a.out / 'results.md').write_text('\n'.join(md))
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
    save(a.out / 'manifest.json', manifest)
    save(a.out / 'progress.json', {'phase': 'complete', 'completed': 15, 'total': 15, 'elapsed_seconds': time.monotonic()-start})
    print('DONE', time.monotonic()-start, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
