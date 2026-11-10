"""DARROW 이력 GBM에 학습 구간의 자기 예측 입력을 한 번 추가한다."""
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e38'))
from history import HistoryDynamics, build_history, initial_history, advance_history, replay, STATE
from surrogate_dw import features
from darrow_env import load_days


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def rollout_examples(dates, days, teacher):
    """12전이마다 실측에서 시작한다. 이후 입력은 예측, 정답만 다음 실측이다."""
    n, members = len(dates), teacher.n_members
    x = np.empty((n, 95, members, 44))
    y = np.empty((n, 95, members))
    member = np.tile(np.arange(members), n)
    for start in range(0, 95, 12):
        state = np.repeat(np.stack([days[d]['state'][start] for d in dates]), members, axis=0)
        history = np.repeat(np.stack([initial_history(days[d]['state'], start) for d in dates]), members, axis=0)
        previous = np.repeat(np.stack([days[d]['zone'][max(start-1, 0)] for d in dates]), members, axis=0)
        for t in range(start, min(start + 12, 95)):
            exo = np.repeat(np.stack([days[d]['exo'][t] for d in dates]), members, axis=0)
            zone = np.repeat(np.stack([days[d]['zone'][t] for d in dates]), members, axis=0)
            xx = np.column_stack((features(state, exo, zone, previous), history.reshape(n * members, 20)))
            target = np.repeat([days[d]['state'][t+1, 0] for d in dates], members)
            x[:, t] = xx.reshape(n, members, 44)
            y[:, t] = (target - state[:, 0]).reshape(n, members)
            nxt = teacher.step(state, exo, zone, previous, member, history)
            history = advance_history(history, state)
            state, previous = nxt, zone
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x.reshape(n * 95, members, 44), y.reshape(n * 95, members)


def confusion(truth, pred):
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    pos, neg = int(truth.sum()), int((~truth).sum())
    return {'tp': tp, 'fp': fp, 'positives': pos, 'negatives': neg,
            'recall_pct': 100 * tp / pos if pos else None,
            'fpr_pct': 100 * fp / neg if neg else None}


def metrics(actual, pred):
    assert actual.shape == (len(pred), pred.shape[2], 5)
    truth = np.broadcast_to(actual[:, None], pred.shape)
    error = np.abs(pred - truth)
    out = {'mae': dict(zip(STATE, error.mean((0, 1, 2)).tolist())),
           'rmse': dict(zip(STATE, np.sqrt(((pred - truth)**2).mean((0, 1, 2))).tolist())),
           'daily_state_mae': error.mean((1, 2)).tolist(), 'confusion': {}, 'counts': {}}
    for name, t, p in (('NH4', truth[..., 0] > 4, pred[..., 0] > 4),
                        ('DO', truth[..., 2:].max(3) > 2, pred[..., 2:].max(3) > 2)):
        out['confusion'][name] = confusion(t, p)
        tc, pc = t.sum(2), p.sum(2)
        day_counts = np.column_stack(((t & p).sum((1, 2)), (~t & p).sum((1, 2)), t.sum((1, 2)), (~t).sum((1, 2))))
        out['confusion'][name]['daily_counts'] = day_counts.tolist()
        out['counts'][name] = {'observed_mean': float(tc.mean()), 'predicted_mean': float(pc.mean()),
                               'mae': float(np.abs(pc - tc).mean()), 'bias': float((pc - tc).mean()),
                               'p95_day_member_absolute_error': float(np.quantile(np.abs(pc - tc), .95))}
    te = np.maximum(truth[..., 0] - 4, 0).sum(2) * .25
    pe = np.maximum(pred[..., 0] - 4, 0).sum(2) * .25
    out['nh4_excess_mg_h_L'] = {'observed_mean': float(te.mean()), 'predicted_mean': float(pe.mean()),
                               'mae': float(np.abs(pe - te).mean())}
    out['nh4_member_mae'] = error[..., 0].mean((0, 2)).tolist()
    out['nh4_p95_daily_mae'] = float(np.quantile(error[..., 0].mean((1, 2)), .95))
    return out


def event_metrics(actual, pred, dq):
    out = []
    for step in (1, 4, 12):
        for group in ('increase', 'decrease', 'decrease_rising'):
            mask = dq > 0 if group == 'increase' else dq < 0
            if group == 'decrease_rising':
                mask = mask & (actual[:, step, 0] - actual[:, 0, 0] > .1)
            a = actual[mask, step, 0]
            p = pred[mask, :, step, 0]
            t = np.broadcast_to(a[:, None], p.shape)
            out.append({'group': group, 'minutes': 15 * step, 'windows': int(mask.sum()),
                        'nh4_mae': float(np.abs(p - t).mean()),
                        'nh4_actual_delta': float((a - actual[mask, 0, 0]).mean()),
                        'nh4_predicted_delta': float((p - actual[mask, 0, 0][:, None]).mean()),
                        'confusion': confusion(t > 4, p > 4)})
    return out


def uncertainty(ref, trial):
    rng = np.random.default_rng(4100)
    starts = rng.integers(40, size=(2000, 6))
    ids = ((starts[:, :, None] + np.arange(7)) % 40).reshape(2000, -1)[:, :40]
    d = np.array(trial['daily_state_mae']) - np.array(ref['daily_state_mae'])
    out = {'mae': {s: {'difference': float(d[:, j].mean()),
                      'paired_7day_95ci': np.quantile(d[ids, j].mean(1), [.025, .975]).tolist()}
                   for j, s in enumerate(STATE)}, 'confusion': {}}
    for name in ('NH4', 'DO'):
        a = np.array(ref['confusion'][name]['daily_counts'])
        b = np.array(trial['confusion'][name]['daily_counts'])
        ar, br = a[ids].sum(1), b[ids].sum(1)
        out['confusion'][name] = {}
        for key, col, den in (('recall_pct', 0, 2), ('fpr_pct', 1, 3)):
            ok = ar[:, den] > 0
            diff = 100 * (br[ok, col] / br[ok, den] - ar[ok, col] / ar[ok, den])
            out['confusion'][name][key] = {'difference_pp': trial['confusion'][name][key] - ref['confusion'][name][key],
                                            'paired_7day_95ci_pp': np.quantile(diff, [.025, .975]).tolist()}
    return out


def screen(ref, trial, ref_events, trial_events):
    checks = {'nh4_mae_lower': trial['mae']['nh4'] < ref['mae']['nh4'],
              'nh4_count_mae_not_higher': trial['counts']['NH4']['mae'] <= ref['counts']['NH4']['mae'],
              'nh4_excess_mae_not_higher': trial['nh4_excess_mg_h_L']['mae'] <= ref['nh4_excess_mg_h_L']['mae']}
    for name in ('NH4', 'DO'):
        checks[name + '_recall_not_lower'] = trial['confusion'][name]['recall_pct'] >= ref['confusion'][name]['recall_pct']
        checks[name + '_fpr_not_higher'] = trial['confusion'][name]['fpr_pct'] <= ref['confusion'][name]['fpr_pct']
    for name in ('do1', 'do2', 'do3'):
        checks[name + '_mae_not_higher'] = trial['mae'][name] <= ref['mae'][name]
    for group in ('increase', 'decrease'):
        for minute in (15, 60, 180):
            find = lambda rows: next(v['nh4_mae'] for v in rows if v['group'] == group and v['minutes'] == minute)
            checks[f'{group}_{minute}min_mae_not_higher'] = find(trial_events) <= find(ref_events)
    return checks


def main(a):
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    source = json.loads((a.source / 'protocol.json').read_text())
    fit, holdout, val = [source[k] for k in ('fit_dates', 'calibration_dates', 'validation_dates')]
    assert (len(fit), len(holdout), len(val)) == (140, 48, 40)
    assert max(fit) < min(holdout) and max(holdout) < min(val)
    protocol = {'locked_at': timestamp(), 'fit_dates': fit, 'holdout_dates': holdout, 'validation_dates': val,
        'hypothesis': 'Training sees measured states only while recursive evaluation sees predicted states. Test whether one round of synthetic recovery examples improves NH4 response without harming event detection or DO.',
        'teacher': 'Frozen e39 history model, fit140 only, all five states recursively predicted. 12-step nonoverlapping within-day windows starting at 0,12,...84, last window11. Measured pre-window history only.',
        'augmentation': 'For each row/member, preserve44 inputs; target is measured next NH4 minus predicted current NH4. Mix observed and synthetic examples1:1. This is trajectory-recovery supervision, not counterfactual physical truth.',
        'arms': 'reference: frozen e39 history; control: observed rows duplicated; augmented: observed plus own member teacher-rollout rows. Same80% row IDs and twice row count, same model structure/monotonicity/seed/iteration budget. Iterations fixed to each e39 history n_iter_, early_stopping=False for both new arms. No search or additional rounds.',
        'scope': 'Only five NH4 GBMs per new arm trained. NO3/DO weights and clipping remain e39. Same recorded future exogenous/actions; no policy training, no test evaluation. DARROW only.',
        'evaluation': 'Primary val40 cold95 replay at fixed predicted NH4>4 and maxDO>2. Descriptive holdout48, teacher-forced one-step and val warm91. Same132 e37 event windows,15/60/180min by airflow direction and observed NH4-rise subgroup.',
        'screen': 'Augmented vs matched control primary: NH4 MAE lower, NH4 count/excess MAE not higher, NH4 and DO recall not lower/FPR not higher, each DO MAE not higher, NH4 event MAE not higher in both directions at15/60/180. Same checks separately vs frozen reference. Zero-tolerance point screen reports all failures, not clinical/practical equivalence; no automatic adoption.',
        'uncertainty': 'Paired7day date-block bootstrap2000 seed4100. Five ensemble members not independent RL seeds. Reused development validation, not confirmatory.',
        'limits': 'No probability/threshold tuning. Actual threshold and costs unchanged. Airflow fixed, no control improvement claim. Synthetic labels are observed next states under recorded actions, not measured responses at perturbed states. One iteration GBM adaptation, no DaD theorem or full algorithm reproduction claim.',
        'source_reference': 'https://ojs.aaai.org/index.php/AAAI/article/view/9590'}
    save(a.out / 'protocol.json', protocol)
    inputs = {str(p): sha(p) for p in (a.prepared / 'data.pkl', a.source / 'models.pkl', a.source / 'predictions.npz', a.source / 'protocol.json', a.source / 'manifest.json', a.events / 'trajectories.npz', a.events / 'results.json')}
    source_manifest = json.loads((a.source / 'manifest.json').read_text())
    assert sha(a.source / 'models.pkl') == source_manifest['model_sha256']
    assert sha(a.prepared / 'data.pkl') == source_manifest['input_sha256'][str(a.prepared / 'data.pkl')]
    for dependency in (HERE.parent / 'e38/history.py', HERE.parent / 'e30/surrogate_dw.py', HERE.parent / 'e30/darrow_env.py'):
        inputs[str(dependency)] = sha(dependency)
    manifest = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'started_at': timestamp(), 'input_sha256': inputs, 'protocol_sha256': sha(a.out / 'protocol.json'),
                'source_sha256': {p.name: sha(p) for p in HERE.glob('*.py')}}
    save(a.out / 'manifest.json', manifest)
    _, days = load_days(a.prepared / 'data.pkl')
    table = pd.read_pickle(a.prepared / 'data.pkl')['data']
    with (a.source / 'models.pkl').open('rb') as f:
        original = pickle.load(f)
    base, teacher_models = original['base'], original['history_nh4']
    teacher = HistoryDynamics(base, teacher_models)
    xt, yt = build_history(table, fit)
    xg, yg = rollout_examples(fit, days, teacher)
    target = (xt[:, 0] + yt[:, 0])[:, None]
    assert np.allclose(xg[:, :, 0] + yg, target, atol=1e-12, rtol=0)
    reset_ids = np.array([day * 95 + t for day in range(140) for t in range(0, 95, 12)])
    assert np.array_equal(xg[reset_ids], np.repeat(xt[reset_ids, None], 5, axis=1))
    np.savez_compressed(a.out / 'training_examples.npz', observed_x=xt, observed_y=yt[:, 0], generated_x=xg, generated_y=yg)
    save(a.out / 'progress.json', {'stage': 'examples_ready', 'elapsed_seconds': time.monotonic()-started})
    print('EXAMPLES', xg.shape, round(time.monotonic()-started, 2), flush=True)
    models = {'reference': teacher_models, 'control': [], 'augmented': []}
    logs, ids_all = [], []
    for member in range(5):
        ids = np.random.RandomState(member).choice(len(xt), int(.8 * len(xt)), replace=False)
        ids_all.append(ids)
        params = teacher_models[member].get_params()
        params.update(max_iter=int(teacher_models[member].n_iter_), early_stopping=False)
        for name in ('control', 'augmented'):
            xx = np.concatenate((xt[ids], xt[ids] if name == 'control' else xg[ids, member]))
            yy = np.concatenate((yt[ids, 0], yt[ids, 0] if name == 'control' else yg[ids, member]))
            begin = time.monotonic()
            model = HistGradientBoostingRegressor(**params).fit(xx, yy)
            models[name].append(model)
            logs.append({'arm': name, 'member': member, 'iterations': model.n_iter_, 'rows': len(xx),
                         'seconds': time.monotonic()-begin, 'row_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest()})
            print('FIT', name, member, model.n_iter_, round(time.monotonic()-started, 2), flush=True)
    np.save(a.out / 'training_row_ids.npy', np.stack(ids_all))
    with (a.out / 'candidate.pkl').open('wb') as f:
        pickle.dump({'schema': 'e41_nh4_history_recovery_requires_history_adapter', 'base': base, 'nh4_models': models}, f)
    manifest['candidate_sha256_before_evaluation'] = sha(a.out / 'candidate.pkl')
    manifest['training_examples_sha256'] = sha(a.out / 'training_examples.npz')
    save(a.out / 'manifest.json', manifest)
    dyns = {name: HistoryDynamics(base, value) for name, value in models.items()}
    ev = np.load(a.events / 'trajectories.npz')
    n = json.loads((a.events / 'results.json').read_text())['support']['val_events']
    windows = [(str(d), int(t)) for d, t in zip(ev['days'][:n], ev['t'][:n])]
    assert n == 132
    xv, yv = build_history(table, val)
    actual_one = (xv[:, :5] + yv).reshape(40, 95, 5)
    old = np.load(a.source / 'predictions.npz')
    arrays = {'event_actual': ev['actual'][:n], 'event_dq': ev['dq'][:n], 'one_step_actual': actual_one}
    results, events = {}, {}
    for name, dyn in dyns.items():
        results[name] = {}
        for mode, dates, start in (('holdout', holdout, 0), ('cold', val, 0), ('warm', val, 4)):
            pred = replay([(d, start) for d in dates], days, dyn, 95-start)[:, :, 1:]
            actual = np.stack([days[d]['state'][start+1:] for d in dates])
            results[name][mode] = metrics(actual, pred)
            arrays[name+'_'+mode], arrays['actual_'+mode] = pred, actual
            if name == 'reference' and mode == 'cold':
                assert np.array_equal(pred, old['history_pred'])
            if name == 'reference' and mode == 'holdout':
                assert np.array_equal(pred, old['history_calibration_pred'])
        one = np.stack([dyn.step(xv[:, :5], xv[:, 5:14], xv[:, 14:19]*1000, xv[:, 19:24]*1000,
                       np.full(len(xv), member), xv[:, 24:].reshape(-1, 4, 5)).reshape(40, 95, 5) for member in range(5)], axis=1)
        arrays[name+'_one_step'] = one
        results[name]['one_step'] = metrics(actual_one, one)
        ep = replay(windows, days, dyn, 12)
        arrays[name+'_events'] = ep
        events[name] = event_metrics(ev['actual'][:n], ep, ev['dq'][:n])
        if name == 'reference':
            assert np.array_equal(ep, old['history_events'])
        print('EVAL', name, json.dumps(results[name]['cold']['mae']), json.dumps(results[name]['cold']['confusion']['NH4'])[:180], flush=True)
    for name in ('control', 'augmented'):
        assert np.array_equal(arrays[name+'_one_step'][..., 1:], arrays['reference_one_step'][..., 1:])
    d, t = windows[0]
    changed = {d: {k: v.copy() for k, v in days[d].items()}}
    changed[d]['state'][t+1:] = 999.
    assert np.array_equal(replay([(d,t)], days, dyns['augmented'], 12), replay([(d,t)], changed, dyns['augmented'], 12))
    lo, hi = xt.copy(), xt.copy()
    lo[:, 14:24] *= .75; hi[:, 14:24] *= 1.25
    monotonicity = {name: [int((m.predict(hi) > m.predict(lo) + 1e-10).sum()) for m in group] for name, group in models.items()}
    assert all(v == [0]*5 for v in monotonicity.values())
    screens = {ref: screen(results[ref]['cold'], results['augmented']['cold'], events[ref], events['augmented']) for ref in ('control', 'reference')}
    uncertainty_result = {ref: uncertainty(results[ref]['cold'], results['augmented']['cold']) for ref in ('control', 'reference')}
    np.savez_compressed(a.out / 'predictions.npz', **arrays)
    save(a.out / 'comparison.json', {'results': results, 'events': events, 'training': logs,
         'screens_augmented_vs': screens, 'all_screen_pass': all(all(v.values()) for v in screens.values()),
         'uncertainty_augmented_minus': uncertainty_result, 'air_monotonicity_wrong_sign_counts': monotonicity,
         'future_actual_scramble_invariance': True, 'no3_do_one_step_identical': True,
         'policy_training': False, 'test_evaluated': False, 'automatic_adoption': False})
    assert all(sha(p) == v for p, v in inputs.items())
    assert sha(a.out / 'protocol.json') == manifest['protocol_sha256']
    assert sha(a.out / 'candidate.pkl') == manifest['candidate_sha256_before_evaluation']
    manifest.update(finished_at=timestamp(), elapsed_seconds=time.monotonic()-started, checks_passed=True)
    save(a.out / 'manifest.json', manifest)
    save(a.out / 'progress.json', {'stage': 'complete', 'elapsed_seconds': manifest['elapsed_seconds']})
    print('DONE', manifest['elapsed_seconds'], json.dumps(screens), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
