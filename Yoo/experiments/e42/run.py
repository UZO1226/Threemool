"""정상/위반 조건부 NH4 예측기를 나누는 한 번의 고정 대조."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from regime import NextNH4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e41'))
import experiment as prior
from history import HistoryDynamics, build_history, initial_history, advance_history, replay
from darrow_env import load_days
from surrogate_dw import STATE


def replay_parts(windows, days, dyn, horizon):
    n, m = len(windows), dyn.n_members
    state = np.repeat(np.stack([days[d]['state'][t] for d,t in windows]), m, axis=0)
    hist = np.repeat(np.stack([initial_history(days[d]['state'], t) for d,t in windows]), m, axis=0)
    previous = np.repeat(np.stack([days[d]['zone'][max(t-1, 0)] for d,t in windows]), m, axis=0)
    member = np.tile(np.arange(m), n)
    states, parts = [state.reshape(n,m,5).copy()], []
    for k in range(horizon):
        exo = np.repeat(np.stack([days[d]['exo'][t+k] for d,t in windows]), m, axis=0)
        zone = np.repeat(np.stack([days[d]['zone'][t+k] for d,t in windows]), m, axis=0)
        nxt = dyn.step(state, exo, zone, previous, member, hist)
        parts.append(np.stack([model.last_parts for model in dyn.history_models], axis=1))
        hist = advance_history(hist, state)
        state, previous = nxt, zone
        states.append(state.reshape(n,m,5).copy())
    return np.stack(states,axis=2), np.stack(parts,axis=2)


def group_error(actual, pred):
    truth = np.broadcast_to(actual[:,None,:,0], pred[...,0].shape)
    result = {}
    for group, mask in (('normal', truth <= 4), ('violation', truth > 4)):
        result[group] = {'samples': int(mask.sum()), 'mae': float(np.abs(pred[...,0][mask]-truth[mask]).mean()),
                         'bias': float((pred[...,0][mask]-truth[mask]).mean()),
                         'predicted_mean': float(pred[...,0][mask].mean()), 'actual_mean': float(truth[mask].mean())}
    return result


def gate_metrics(actual, parts):
    truth = np.broadcast_to(actual[:,None,:,0] > 4, parts[...,0].shape)
    p = parts[...,0]
    return {'auc': float(roc_auc_score(truth.ravel(),p.ravel())),
            'brier': float(brier_score_loss(truth.ravel(),p.ravel())),
            'mean_probability_normal': float(p[~truth].mean()), 'mean_probability_violation': float(p[truth].mean()),
            'mean_normal_expert_on_violations': float(parts[...,1][truth].mean()),
            'mean_violation_expert_on_violations': float(parts[...,2][truth].mean()),
            'probability05_diagnostic_only': prior.confusion(truth,p>.5)}


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True,exist_ok=False)
    p41 = json.loads((a.source/'protocol.json').read_text())
    protocol = {k:p41[k] for k in ('fit_dates','holdout_dates','validation_dates')}
    protocol.update(locked_at=prior.timestamp(),
        hypothesis='Separate normal and violating conditional next-concentration models to prevent majority normal examples from directly fitting the high-concentration expert.',
        data='Reuse exact e41 observed+synthetic inputs1:1, fixed80% row IDs, five members,44 features. No new augmentation/weight/threshold sweep. Fit140 only.',
        model='Classifier g=P(actual_next_NH4>4|x) with unweighted binary log loss; normal regressor on target<=4; violation regressor on target>4. Both squared error. Predict (1-g)*clip(mu0,0,4)+g*clip(mu1,4,fit_hi). All use same inherited air monotone constraints. No true label at inference.',
        arms='Existing e41 control and augmented as references; direct: single absolute-next regressor T trees; capacity: single absolute-next regressor3T trees; split: gate/normal/violation eachT trees. T=e41 member n_iter_, no early stopping. 3T matches sum of tree counts, not row-tree work or wall time; disclose both.',
        contrasts='Primary split vs direct and capacity isolate regime split from target parameterization and tree count. Compare against e41 control for practical retention. e41 augmented quantifies normal-error/FPR improvements to preserve.',
        evaluation='Fixed actual/predicted NH4>4 and maxDO>2. Val40 cold95 primary, holdout48/warm91/one-step descriptive. Same132 event windows15/60/180 and rising subgroup. No inference sampling or separate gate-cost definition.',
        screen='e41 sixteen-point screen split against direct/capacity/control separately, plus normal-group MAE and FPR no higher than e41 augmented, violation-group MAE no higher than e41 control. Zero-tolerance point screen, not practical equivalence. Report all changes and paired7day CI2000 seed4100.',
        limits='Expected concentration can still suppress rare peaks. Gate probability may be wrong under shift. Training groups use labels only for fitting; prediction sees current/past simulated states and recorded exogenous/actions. Synthetic recovery labels not counterfactual physics. Reused development validation, no independent confirmation. No new RL/test or automatic adoption.',
        sources=['https://www.cs.toronto.edu/~hinton/absps/jjnh91.html','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html'])
    prior.save(a.out/'protocol.json',protocol)
    inputs = {str(path):prior.sha(path) for path in (a.prepared/'data.pkl',a.source/'candidate.pkl',a.source/'training_examples.npz',a.source/'training_row_ids.npy',a.source/'predictions.npz',a.source/'comparison.json',a.source/'manifest.json',a.events/'trajectories.npz',a.events/'results.json',HERE.parent/'e41/experiment.py',HERE.parent/'e38/history.py',HERE.parent/'e30/surrogate_dw.py',HERE.parent/'e30/darrow_env.py')}
    old_manifest = json.loads((a.source/'manifest.json').read_text())
    assert prior.sha(a.source/'candidate.pkl') == old_manifest['candidate_sha256_before_evaluation']
    assert prior.sha(a.source/'training_examples.npz') == old_manifest['training_examples_sha256']
    assert prior.sha(a.prepared/'data.pkl') == old_manifest['input_sha256'][str(a.prepared/'data.pkl')]
    manifest = {'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'started_at':prior.timestamp(),
                'input_sha256':inputs, 'source_sha256':{f.name:prior.sha(f) for f in HERE.glob('*.py')},
                'protocol_sha256':prior.sha(a.out/'protocol.json')}
    prior.save(a.out/'manifest.json',manifest)
    with (a.source/'candidate.pkl').open('rb') as f:
        old = pickle.load(f)
    base = old['base']
    examples = np.load(a.source/'training_examples.npz')
    ids = np.load(a.source/'training_row_ids.npy')
    models = {'control':old['nh4_models']['control'], 'augmented':old['nh4_models']['augmented'],
              'direct':[], 'capacity':[], 'split':[]}
    logs = []
    for m in range(5):
        x0 = examples['observed_x'][ids[m]]
        x1 = examples['generated_x'][ids[m],m]
        y0 = x0[:,0]+examples['observed_y'][ids[m]]
        y1 = x1[:,0]+examples['generated_y'][ids[m],m]
        assert np.allclose(y0,y1,atol=1e-12,rtol=0)
        x, y = np.concatenate((x0,x1)), np.concatenate((y0,y0))
        # Use the original measured next value for both copies, avoiding floating-point boundary drift.
        labels = y > 4
        params = old['nh4_models']['augmented'][m].get_params()
        T = int(old['nh4_models']['augmented'][m].n_iter_)
        assert params['early_stopping'] is False and params['max_iter']==T
        for name,factor in (('direct',1),('capacity',3)):
            begin = time.monotonic()
            cfg = dict(params,max_iter=factor*T)
            fitted = HistGradientBoostingRegressor(**cfg).fit(x,y)
            models[name].append(NextNH4(base['hi'][0],single=fitted))
            logs.append(dict(arm=name,member=m,trees=fitted.n_iter_,rows=len(x),row_tree_work=len(x)*fitted.n_iter_,seconds=time.monotonic()-begin))
        begin = time.monotonic()
        gate_cfg = {k:v for k,v in params.items() if k not in ('loss','quantile')}
        gate = HistGradientBoostingClassifier(loss='log_loss',class_weight=None,**gate_cfg).fit(x,labels)
        normal = HistGradientBoostingRegressor(**params).fit(x[~labels],y[~labels])
        violation = HistGradientBoostingRegressor(**params).fit(x[labels],y[labels])
        models['split'].append(NextNH4(base['hi'][0],gate=gate,normal=normal,violation=violation))
        logs.append(dict(arm='split',member=m,trees=gate.n_iter_+normal.n_iter_+violation.n_iter_,rows=len(x),
                         positive_rows=int(labels.sum()),positive_unique_rows=int(labels[:len(y0)].sum()),
                         positive_dates=int(len(np.unique((ids[m]//95)[labels[:len(y0)]]))),
                         row_tree_work=2*len(x)*T,seconds=time.monotonic()-begin))
        print('FIT member',m,'T',T,'positive',labels.sum(),'elapsed',round(time.monotonic()-start,2),flush=True)
        prior.save(a.out/'progress.json',{'stage':'fit','members_complete':m+1,'members_total':5,'elapsed_seconds':time.monotonic()-start})
    with (a.out/'candidate.pkl').open('wb') as f:
        pickle.dump({'schema':'e42_conditional_next_nh4_requires_history_adapter','base':base,'nh4_models':models},f)
    manifest['candidate_sha256_before_eval'] = prior.sha(a.out/'candidate.pkl')
    prior.save(a.out/'manifest.json',manifest)
    _, days = load_days(a.prepared/'data.pkl')
    table = pd.read_pickle(a.prepared/'data.pkl')['data']
    val, holdout = protocol['validation_dates'],protocol['holdout_dates']
    xv,yv = build_history(table,val)
    actual_one = (xv[:,:5]+yv).reshape(40,95,5)
    ev = np.load(a.events/'trajectories.npz')
    n = json.loads((a.events/'results.json').read_text())['support']['val_events']
    windows = [(str(d),int(t)) for d,t in zip(ev['days'][:n],ev['t'][:n])]
    assert n==132
    previous = np.load(a.source/'predictions.npz')
    results, events, groups, gates = {},{},{},{}
    arrays = {'event_actual':ev['actual'][:n],'event_dq':ev['dq'][:n],'one_step_actual':actual_one}
    dyns = {k:HistoryDynamics(base,v) for k,v in models.items()}
    for name,dyn in dyns.items():
        results[name],groups[name] = {},{}
        for mode,dates,t0 in (('holdout',holdout,0),('cold',val,0),('warm',val,4)):
            ww = [(d,t0) for d in dates]
            if name=='split':
                pred,parts = replay_parts(ww,days,dyn,95-t0)
                arrays['split_parts_'+mode] = parts
            else:
                pred = replay(ww,days,dyn,95-t0)
            pred = pred[:,:,1:]
            actual = np.stack([days[d]['state'][t0+1:] for d in dates])
            arrays[name+'_'+mode],arrays['actual_'+mode] = pred,actual
            results[name][mode] = prior.metrics(actual,pred)
            groups[name][mode] = group_error(actual,pred)
            if name=='split':
                gates[mode] = gate_metrics(actual,parts)
                assert np.allclose(pred[...,0],(1-parts[...,0])*parts[...,1]+parts[...,0]*parts[...,2],atol=1e-12,rtol=0)
            if name in ('control','augmented'):
                assert np.array_equal(pred,previous[name+'_'+mode])
        one, one_parts = [],[]
        for m in range(5):
            one.append(dyn.step(xv[:,:5],xv[:,5:14],xv[:,14:19]*1000,xv[:,19:24]*1000,np.full(len(xv),m),xv[:,24:].reshape(-1,4,5)).reshape(40,95,5))
            if name=='split':one_parts.append(models[name][m].last_parts.reshape(40,95,3))
        one = np.stack(one,axis=1)
        arrays[name+'_one_step'] = one
        results[name]['one_step'] = prior.metrics(actual_one,one)
        groups[name]['one_step'] = group_error(actual_one,one)
        if name=='split':
            parts = np.stack(one_parts,axis=1); arrays['split_parts_one_step'] = parts
            gates['one_step'] = gate_metrics(actual_one,parts)
            ep,parts = replay_parts(windows,days,dyn,12);arrays['split_parts_events'] = parts
        else:
            ep = replay(windows,days,dyn,12)
        arrays[name+'_events'] = ep
        events[name] = prior.event_metrics(ev['actual'][:n],ep,ev['dq'][:n])
        if name in ('control','augmented'):assert np.array_equal(ep,previous[name+'_events'])
        print('EVAL',name,results[name]['cold']['mae']['nh4'],results[name]['cold']['confusion']['NH4']['recall_pct'],results[name]['cold']['confusion']['NH4']['fpr_pct'],flush=True)
    for name in models:
        assert np.array_equal(arrays[name+'_one_step'][...,1:],arrays['control_one_step'][...,1:])
    d,t = windows[0]
    changed = {d:{key:value.copy() for key,value in days[d].items()}}
    changed[d]['state'][t+1:] = 999.
    expected = replay([(d,t)],days,dyns['split'],12)
    assert np.array_equal(expected,replay([(d,t)],changed,dyns['split'],12))
    traced,_ = replay_parts([(d,t)],days,dyns['split'],12)
    assert np.array_equal(expected,traced)
    x = examples['observed_x'];lo,hi = x.copy(),x.copy();lo[:,14:24]*=.75;hi[:,14:24]*=1.25
    monotonicity = {name:[int((model.predict(hi)>model.predict(lo)+1e-10).sum()) for model in group] for name,group in models.items()}
    assert all(counts==[0]*5 for counts in monotonicity.values())
    screens = {ref:prior.screen(results[ref]['cold'],results['split']['cold'],events[ref],events['split']) for ref in ('control','direct','capacity')}
    preservation = {'normal_mae_not_above_augmented':groups['split']['cold']['normal']['mae']<=groups['augmented']['cold']['normal']['mae'],
                    'fpr_not_above_augmented':results['split']['cold']['confusion']['NH4']['fpr_pct']<=results['augmented']['cold']['confusion']['NH4']['fpr_pct'],
                    'violation_mae_not_above_control':groups['split']['cold']['violation']['mae']<=groups['control']['cold']['violation']['mae']}
    uncertainty = {ref:prior.uncertainty(results[ref]['cold'],results['split']['cold']) for ref in ('control','direct','capacity')}
    prior.save(a.out/'comparison.json',{'results':results,'events':events,'group_errors':groups,'gate_diagnostics':gates,'training':logs,
               'screens_split_vs':screens,'preservation_screen':preservation,'all_screen_pass':all(all(v.values()) for v in screens.values()) and all(preservation.values()),
               'uncertainty_split_minus':uncertainty,'air_monotonicity_wrong_sign_counts':monotonicity,
               'future_actual_invariance':True,'NO3_DO_weights_unchanged':True,'no_RL_or_test':True,'automatic_adoption':False})
    np.savez_compressed(a.out/'predictions.npz',**arrays)
    assert all(prior.sha(path)==digest for path,digest in inputs.items())
    assert prior.sha(a.out/'protocol.json')==manifest['protocol_sha256']
    assert prior.sha(a.out/'candidate.pkl')==manifest['candidate_sha256_before_eval']
    manifest.update(finished_at=prior.timestamp(),elapsed_seconds=time.monotonic()-start,checks_passed=True)
    prior.save(a.out/'manifest.json',manifest)
    prior.save(a.out/'progress.json',{'stage':'complete','elapsed_seconds':manifest['elapsed_seconds']})
    print('DONE',manifest['elapsed_seconds'],json.dumps(preservation),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
