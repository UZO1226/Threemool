"""같은 표본과 GBM 설정으로 NH4 이력 유무만 대조한다."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from history import HistoryDynamics, build_history, initial_history, advance_history, replay, STATE
from common import sha, save
from darrow_env import load_days

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e35'))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

metric = module('e35_metrics', HERE.parent/'e35/run.py')
event = module('e37_events', HERE.parent/'e37/run.py')


def summary(actual, pred):
    n, m, t, k = pred.shape
    return metric.summarize(np.repeat(actual, m, axis=0), pred.reshape(n*m, t, k))


def paired_day_ci(a, b, actual):
    delta = (np.abs(b-actual[:, None])-np.abs(a-actual[:, None])).mean((1, 2))
    rng = np.random.default_rng(3800)
    starts = rng.integers(40, size=(2000, 6))
    ids = ((starts[:, :, None]+np.arange(7)) % 40).reshape(2000, -1)[:, :40]
    boot = delta[ids].mean(1)
    return {s: {'history_minus_base_mae': float(delta[:, j].mean()),
                'paired_7day_block_95ci': np.quantile(boot[:, j], [.025, .975]).tolist()}
            for j,s in enumerate(STATE)}


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    splits, days = load_days(a.prepared/'data.pkl')
    data = pd.read_pickle(a.prepared/'data.pkl')
    with a.base.open('rb') as f:
        base = pickle.load(f)
    protocol = {'locked_at': datetime.now(timezone.utc).isoformat(),
                'changes': 'NH4 GBM only: 4 lags x 5 water states added to existing 24 inputs (44 total). No architecture or hyperparameter sweep.',
                'history_order': ['15min', '30min', '45min', '60min'], 'state_order': STATE,
                'padding': 'Within-day unavailable past repeats the first state, in training and cold-start evaluation.',
                'training': 'same 188 dates, all 95 transitions/day, same 80% row samples and GBM parameters/early-stopping RNG per member. NH4 unchanged-model refit must reproduce baseline exactly.',
                'fixed': 'NO3 and DO model weights, clipping, exogenous inputs, recorded actions, policy, reward and constraints unchanged.',
                'validation': '40 dates; one-step teacher forcing descriptive, primary cold 95-step replay, secondary observed first-hour warm start then 91 free steps. No cross-mode comparisons.',
                'events': 'Same 132 e37 windows and 15/60/180min; only pre-start measured history, then predicted history.',
                'screen': 'Primary NH4 rollout MAE lower, NH4 recall not lower/FPR not higher; all three DO MAEs not higher, DO recall not lower/FPR not higher; NH4 response MAE not higher for both action directions at all three horizons. Zero tolerance point-estimate screen, no success guarantee; report paired uncertainty and all failures. Passing would still not certify causal control effects.',
                'test_evaluated': False, 'RL_training': False, 'no_automatic_adoption': True}
    save(a.out/'protocol.json', protocol)
    inputs = {str(p): sha(p) for p in (a.prepared/'data.pkl', a.base, a.events/'trajectories.npz', a.events/'results.json')}
    manifest = {'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'source_sha256': {p.name: sha(p) for p in HERE.glob('*.py')},
                'input_sha256': inputs, 'protocol_sha256': sha(a.out/'protocol.json'), 'started_at': datetime.now(timezone.utc).isoformat()}
    save(a.out/'manifest.json', manifest)
    xt, yt = build_history(data['data'], splits['train'])
    xv, yv = build_history(data['data'], splits['val'])
    assert xt.shape == (188*95, 44) and xv.shape == (40*95, 44)
    frozen = [[pickle.dumps(model) for model in row] for row in base['models']]
    history_models, log = [], []
    for m in range(5):
        ids = np.random.RandomState(m).choice(len(xt), int(.8*len(xt)), replace=False)
        params = base['models'][m][0].get_params()
        replica = HistGradientBoostingRegressor(**params).fit(xt[ids, :24], yt[ids, 0])
        assert np.array_equal(replica.predict(xv[:, :24]), base['models'][m][0].predict(xv[:, :24]))
        params['monotonic_cst'] = np.r_[params['monotonic_cst'], np.zeros(20, dtype=int)]
        model = HistGradientBoostingRegressor(**params).fit(xt[ids], yt[ids, 0])
        history_models.append(model)
        log.append({'member':m, 'iterations':model.n_iter_, 'base_refit_exact':True})
        print('FIT', m, model.n_iter_, f'{time.monotonic()-start:.2f}s', flush=True)
    with (a.out/'candidate.pkl').open('wb') as f:
        pickle.dump({'schema':'nh4_history_v1_requires_history_adapter', 'base_model':base, 'nh4_history_models':history_models}, f)
    dyns = {'base':HistoryDynamics(base), 'history':HistoryDynamics(base, history_models)}
    results, arrays = {}, {}
    event_data = np.load(a.events/'trajectories.npz')
    er = json.loads((a.events/'results.json').read_text())
    n = er['support']['val_events']
    ew = [(str(d), int(t)) for d,t in zip(event_data['days'][:n], event_data['t'][:n])]
    assert n == 132
    for name,dyn in dyns.items():
        one = []
        for m in range(5):
            one.append(dyn.step(xv[:, :5], xv[:, 5:14], xv[:, 14:19]*1000, xv[:, 19:24]*1000,
                                np.full(len(xv),m), xv[:, 24:].reshape(-1,4,5)))
        one = np.stack(one)
        truth = xv[:, :5]+yv
        rr = {'one_step_ensemble_mean':metric.summarize(truth[None], one.mean(0)[None])}
        arrays[name+'_one_step'] = one
        for label, t0 in [('cold',0), ('warm',4)]:
            windows = [(d,t0) for d in splits['val']]
            pred = replay(windows, days, dyn, 95-t0)[:, :, 1:]
            actual = np.stack([days[d]['state'][t0+1:] for d in splits['val']])
            arrays[name+'_'+label], arrays['actual_'+label] = pred, actual
            rr[label] = summary(actual, pred)
            rr[label]['nh4_excess_mg_h_L_predicted'] = float((np.maximum(pred[:,:,:,0]-4,0).sum(2)*.25).mean())
            rr[label]['nh4_excess_mg_h_L_observed'] = float((np.maximum(actual[:,:,0]-4,0).sum(1)*.25).mean())
        ep = replay(ew, days, dyn, 12)
        if name == 'base':
            assert np.array_equal(ep, event_data['pred'][:n])
        arrays[name+'_events'] = ep
        mask = event_data['dq'][:n] > 0
        wl = [{'day':d,'t':t} for d,t in ew]
        rr['events'] = {label:event.summarize(event_data['actual'][:n], ep, selector, wl, splits['val'])
                        for label,selector in [('all',np.ones(n,bool)),('increase',mask),('decrease',~mask)]}
        results[name] = rr
        print('EVAL', name, json.dumps(rr['cold']), flush=True)
    arrays['actual_events'] = event_data['actual'][:n]
    arrays['event_dq'] = event_data['dq'][:n]
    arrays['one_step_actual'] = truth
    np.savez_compressed(a.out/'predictions.npz', **arrays)
    baseline = json.loads((HERE.parent/'e35/comparison.json').read_text())['results']['corrected']['rollout_member_average']
    assert results['base']['cold']['mae'] == baseline['mae']
    assert np.array_equal(arrays['base_one_step'][:,:,1:], arrays['history_one_step'][:,:,1:])
    for m,row in enumerate(base['models']):
        for j,model in enumerate(row):
            assert pickle.dumps(model) == frozen[m][j]
    # Scrambling post-start measured states cannot affect recursive predictions.
    d,t = ew[0]
    changed = {d:{k:v.copy() for k,v in days[d].items()}}
    changed[d]['state'][t+1:] = 999.
    assert np.array_equal(replay([(d,t)],days,dyns['history'],12), replay([(d,t)],changed,dyns['history'],12))
    # Fixed historical states and other features preserve the inherited NH4 air sign constraint.
    lo,hi = xv.copy(), xv.copy()
    lo[:,14:24] *= .75
    hi[:,14:24] *= 1.25
    wrong_sign = [int((model.predict(hi)>model.predict(lo)+1e-10).sum()) for model in history_models]
    assert wrong_sign == [0]*5
    b,h = results['base']['cold'],results['history']['cold']
    screen = {'nh4_mae_lower': h['mae']['nh4'] < b['mae']['nh4']}
    for target in ('NH4','DO'):
        screen[target+'_recall_not_lower'] = h['confusion'][target]['recall_pct'] >= b['confusion'][target]['recall_pct']
        screen[target+'_fpr_not_higher'] = h['confusion'][target]['fpr_pct'] <= b['confusion'][target]['fpr_pct']
    for target in ('do1','do2','do3'):
        screen[target+'_mae_not_higher'] = h['mae'][target] <= b['mae'][target]
    for group in ('increase','decrease'):
        for time_min in (15,60,180):
            def find(name):
                return next(r['mae'] for r in results[name]['events'][group]['metrics'] if r['target']=='nh4' and r['minutes']==time_min)
            screen[f'{group}_{time_min}min_nh4_mae_not_higher'] = find('history') <= find('base')
    result = {'results':results, 'training':log, 'screen':screen, 'all_screen_checks_passed':all(screen.values()),
              'cold_paired_uncertainty':paired_day_ci(arrays['base_cold'],arrays['history_cold'],arrays['actual_cold']),
              'warm_paired_uncertainty':paired_day_ci(arrays['base_warm'],arrays['history_warm'],arrays['actual_warm']),
              'air_monotonicity_wrong_sign_counts':wrong_sign, 'base_models_unchanged':True,
              'post_start_truth_scramble_invariance':True, 'policy_training':False, 'test_evaluated':False,
              'no_automatic_adoption':True}
    save(a.out/'comparison.json',result)
    assert all(sha(Path(p)) == v for p,v in inputs.items())
    assert sha(a.out/'protocol.json') == manifest['protocol_sha256']
    manifest.update(elapsed_seconds=time.monotonic()-start, checks_passed=True, finished_at=datetime.now(timezone.utc).isoformat(),
                    candidate_sha256=sha(a.out/'candidate.pkl'))
    save(a.out/'manifest.json',manifest)
    print('DONE',manifest['elapsed_seconds'],json.dumps(screen),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
