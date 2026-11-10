"""모델 학습 날짜와 분리한 DARROW 날짜로 NH4 위험 판정을 보정한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,brier_score_loss

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e38'))
from history import HistoryDynamics,build_history,replay,STATE
from common import save,sha
from darrow_env import load_days
sys.path.insert(0,str(HERE.parent/'e35'))
from mapping import monotone_constraints


def choose(prob,truth,mode):
    fpr,tpr,threshold=roc_curve(truth,prob,drop_intermediate=False)
    threshold=np.where(np.isfinite(threshold),threshold,np.nextafter(1.,2.))
    if mode=='recall80':
        i=int(np.flatnonzero(tpr>=.8)[0])
    elif mode=='fpr10':
        i=int(np.flatnonzero(fpr<=.1)[-1])
    else:
        raise ValueError(mode)
    return {'probability_threshold':float(threshold[i]),'calibration_recall_pct':float(100*tpr[i]),
            'calibration_fpr_pct':float(100*fpr[i]),'rule':'>='}


def classify(truth,alerts):
    assert truth.shape==alerts.shape
    axes=tuple(range(1,truth.ndim))
    tp=(truth&alerts).sum(axes);fp=(~truth&alerts).sum(axes)
    pos=truth.sum(axes);neg=(~truth).sum(axes)
    return {'recall_pct':float(100*tp.sum()/pos.sum()),'fpr_pct':float(100*fp.sum()/neg.sum()),
            'tp':int(tp.sum()),'fp':int(fp.sum()),'positives':int(pos.sum()),'negatives':int(neg.sum()),
            'predicted_count_per_day_member':float(alerts.sum(-1).mean()),
            'observed_count_per_day_member':float(truth.sum(-1).mean()),
            'day_counts':np.column_stack([tp,fp,pos,neg]).tolist()}


def pair_ci(a,b):
    a,b=np.array(a['day_counts']),np.array(b['day_counts'])
    assert len(a)==len(b)==40 and np.array_equal(a[:,2:],b[:,2:])
    rng=np.random.default_rng(3900);starts=rng.integers(40,size=(2000,6))
    ids=((starts[:,:,None]+np.arange(7))%40).reshape(2000,-1)[:,:40]
    aa,bb=a[ids].sum(1),b[ids].sum(1)
    good=(aa[:,2]>0)&(aa[:,3]>0)
    out={}
    for key,j,d in [('recall_pct',0,2),('fpr_pct',1,3)]:
        delta=100*(bb[good,j]/bb[good,d]-aa[good,j]/aa[good,d])
        out[key]={'difference_pp':float(100*(b[:,j].sum()/b[:,d].sum()-a[:,j].sum()/a[:,d].sum())),
                  'paired_7day_block_95ci_pp':np.quantile(delta,[.025,.975]).tolist()}
    return out


def evaluate(actual,pred,calibrator,operating):
    truth=np.repeat(actual[:,None,:,0]>4,5,axis=1)
    score=pred[:,:,:,0]
    prob=calibrator.predict(score.ravel()).reshape(score.shape)
    labels={'raw':score>4}
    labels.update({key:prob>=v['probability_threshold'] for key,v in operating.items()})
    result={'nh4':{key:classify(truth,x) for key,x in labels.items()},
            'mae':dict(zip(STATE,np.abs(pred-actual[:,None]).mean((0,1,2)).tolist())),
            'DO':classify(np.repeat(actual[:,None,:,2:].max(3)>2,5,axis=1),pred[:,:,:,2:].max(3)>2),
            'raw_score_auc':float(roc_auc_score(truth.ravel(),score.ravel())),
            'calibrated_auc':float(roc_auc_score(truth.ravel(),prob.ravel())),
            'calibrated_average_precision':float(average_precision_score(truth.ravel(),prob.ravel())),
            'brier':float(brier_score_loss(truth.ravel(),prob.ravel()))}
    return result,prob,labels


def main(a):
    start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    splits,days=load_days(a.prepared/'data.pkl');data=pd.read_pickle(a.prepared/'data.pkl')
    fit,cal,val=splits['train'][:140],splits['train'][140:],splits['val']
    assert (len(fit),len(cal),len(val))==(140,48,40)
    assert max(fit)<min(cal) and max(cal)<min(val)
    assert not(set(fit)&set(cal) or set(cal)&set(val))
    protocol={'locked_at':datetime.now(timezone.utc).isoformat(), 'fit_dates':fit,'calibration_dates':cal,'validation_dates':val,
              'model': 'All 25 base GBMs refit on first140 dates, corrected DO constraints. Five history NH4 GBMs trained on identical row samples; other20 models shared. Clipping from fit140 only.',
              'calibration':'Isotonic regression, increasing, out_of_bounds=clip. Input only recursively predicted NH4 concentration, label actual NH4>4. Fit on held-out-from-dynamics 48 dates. Equal date/member/sample weighting.',
              'primary_operating_rule':'On calibration only, select highest probability threshold attaining recall>=80% (minimal FPR among such threshold points). Frozen before validation.',
              'secondary_operating_rule':'On calibration only, select threshold with maximal recall at FPR<=10%. No validation-dependent selection between profiles.',
              'raw_rule':'predicted NH4>4. True violation remains actual NH4>4. Calibration changes alert classification only, never physical concentration, recursive trajectory, reward or constraints.',
              'comparisons':'base/raw, history/raw, base/calibrated, history/calibrated; report primary recall80 and separately secondary fpr10. Within-profile achieved validation recall may differ: do not call it matched recall.',
              'screen':'History/calibrated primary validation recall>=base/calibrated AND FPR<=base/calibrated, at least one strict; also report whether validation meets calibration recall80 target. No causal/controller adoption.',
              'events':'Same e37 windows: factual NH4 growth >0.1 mg/L, rising windows following airflow decrease at15/60/180min. Frozen calibrator applied without event tuning; labels actual NH4>4, not growth itself.',
              'uncertainty':'Paired dates, circular7dayblock bootstrap2000. Development evaluation, no independent final-test or ensemble-as-independent-seed inference.',
              'exogenous':'Inherited DARROW prepared inputs include interpolated synthetic influent; recorded future actions/exogenous inputs used for conditional replay, not online forecasting.',
              'RL_training':False,'test_evaluated':False}
    save(a.out/'protocol.json',protocol)
    inputs={str(p):sha(p) for p in (a.prepared/'data.pkl',a.events/'trajectories.npz',a.events/'results.json')}
    manifest={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'input_sha256':inputs,'source_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
              'protocol_sha256':sha(a.out/'protocol.json'),'started_at':datetime.now(timezone.utc).isoformat()}
    save(a.out/'manifest.json',manifest)
    xt,yt=build_history(data['data'],fit)
    assert xt.shape==(140*95,44)
    all_fit=np.concatenate([days[d]['state'] for d in fit])
    base={'models':[],'hi':all_fit.max(0)*1.1};hist=[];training=[]
    for m in range(5):
        ids=np.random.RandomState(m).choice(len(xt),int(.8*len(xt)),replace=False)
        row=[]
        for j in range(5):
            params=dict(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,
                        early_stopping=True,validation_fraction=.1,n_iter_no_change=20,random_state=m+j)
            g=HistGradientBoostingRegressor(**params,monotonic_cst=monotone_constraints(j)).fit(xt[ids,:24],yt[ids,j]);row.append(g)
            training.append({'member':m,'target':STATE[j],'history':False,'iterations':g.n_iter_})
            if j==0:
                h=HistGradientBoostingRegressor(**params,monotonic_cst=np.r_[monotone_constraints(j),np.zeros(20,dtype=int)]).fit(xt[ids],yt[ids,j]);hist.append(h)
                training.append({'member':m,'target':'nh4','history':True,'iterations':h.n_iter_})
        base['models'].append(row)
        save(a.out/'progress.json',{'stage':'fit','models_done':len(training),'models_total':30,'elapsed_seconds':time.monotonic()-start})
        print('FIT',len(training),'/30',round(time.monotonic()-start,1),flush=True)
    dyns={'base':HistoryDynamics(base),'history':HistoryDynamics(base,hist)}
    with (a.out/'models.pkl').open('wb') as f:pickle.dump({'base':base,'history_nh4':hist},f)
    model_hash=sha(a.out/'models.pkl')
    calibrators,operating,cal_predictions,cal_stats={},{},{},{}
    actual_cal=np.stack([days[d]['state'][1:] for d in cal])
    ycal=np.repeat(actual_cal[:,None,:,0]>4,5,axis=1)
    for name,dyn in dyns.items():
        pred=replay([(d,0) for d in cal],days,dyn,95)[:,:,1:];cal_predictions[name]=pred
        scores=pred[:,:,:,0]
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(scores.ravel(),ycal.ravel().astype(float))
        probabilities=iso.predict(scores.ravel())
        op={mode:choose(probabilities,ycal.ravel(),mode) for mode in ('recall80','fpr10')}
        calibrators[name],operating[name]=iso,op
        cal_stats[name]={'raw':classify(ycal,scores>4),'operating_points':op,'auc':float(roc_auc_score(ycal.ravel(),scores.ravel()))}
        print('CALIBRATED',name,json.dumps(op),flush=True)
    save(a.out/'operating_points.json',{'locked_at':datetime.now(timezone.utc).isoformat(),'models':operating,'calibration_metrics':cal_stats})
    with (a.out/'calibrators.pkl').open('wb') as f:pickle.dump(calibrators,f)
    manifest.update(operating_sha256_before_validation=sha(a.out/'operating_points.json'),calibrators_sha256_before_validation=sha(a.out/'calibrators.pkl'),model_sha256=model_hash)
    save(a.out/'manifest.json',manifest)
    actual=np.stack([days[d]['state'][1:] for d in val])
    results,arrays={}, {'actual':actual,'actual_calibration':actual_cal}
    ev=np.load(a.events/'trajectories.npz');n=json.loads((a.events/'results.json').read_text())['support']['val_events']
    windows=[(str(d),int(t)) for d,t in zip(ev['days'][:n],ev['t'][:n])]
    event_summary=[]
    for name,dyn in dyns.items():
        pred=replay([(d,0) for d in val],days,dyn,95)[:,:,1:]
        result,prob,labels=evaluate(actual,pred,calibrators[name],operating[name]);results[name]=result
        arrays[name+'_pred'],arrays[name+'_prob'],arrays[name+'_calibration_pred']=pred,prob,cal_predictions[name]
        for key,x in labels.items():arrays[name+'_alert_'+key]=x
        ep=replay(windows,days,dyn,12);arrays[name+'_events']=ep
        for horizon in (1,4,12):
            for group in ('all','air_decrease_and_nh4_rising'):
                mask=np.ones(n,bool) if group=='all' else (ev['dq'][:n]<0)&(ev['actual'][:n,horizon,0]-ev['actual'][:n,0,0]>.1)
                if not mask.any():continue
                tr=np.repeat((ev['actual'][:n,horizon,0][mask]>4)[:,None,None],5,axis=1)
                score=ep[mask,:,horizon,0][:,:,None];pp=calibrators[name].predict(score.ravel()).reshape(score.shape)
                alerts={'raw':score>4};alerts.update({key:pp>=v['probability_threshold'] for key,v in operating[name].items()})
                for mode,aa in alerts.items():
                    # Sparse event groups may contain no positives or negatives.
                    positives=int(tr.sum());negatives=int((~tr).sum())
                    tp=int((tr&aa).sum());fp=int((~tr&aa).sum())
                    event_summary.append({'model':name,'mode':mode,'group':group,'minutes':horizon*15,'windows':int(mask.sum()),
                                          'positives':positives,'negatives':negatives,'recall_pct':100*tp/positives if positives else None,
                                          'fpr_pct':100*fp/negatives if negatives else None,'tp':tp,'fp':fp})
        print('VAL',name,json.dumps({k:{metric:q[metric] for metric in ('recall_pct','fpr_pct')} for k,q in result['nh4'].items()}),flush=True)
    arrays['event_actual']=ev['actual'][:n];arrays['event_dq']=ev['dq'][:n]
    np.savez_compressed(a.out/'predictions.npz',**arrays)
    diffs={mode:pair_ci(results['base']['nh4'][mode],results['history']['nh4'][mode]) for mode in ('raw','recall80','fpr10')}
    within={name:pair_ci(results[name]['nh4']['raw'],results[name]['nh4']['recall80']) for name in dyns}
    b,h=results['base']['nh4']['recall80'],results['history']['nh4']['recall80']
    screen={'history_calibrated_recall_not_lower':h['recall_pct']>=b['recall_pct'],
            'history_calibrated_fpr_not_higher':h['fpr_pct']<=b['fpr_pct'],
            'at_least_one_strict':h['recall_pct']>b['recall_pct'] or h['fpr_pct']<b['fpr_pct'],
            'validation_recall80_reached':h['recall_pct']>=80}
    result={'validation':results,'training':training,'event_diagnostics':event_summary,'history_minus_base':diffs,
            'calibrated_minus_raw_within_model':within,'screen':screen,'all_screen_pass':all(screen.values()),
            'dates':{'fit':len(fit),'calibration':len(cal),'val':len(val)},'no_automatic_adoption':True,
            'physical_truth_threshold_unchanged':4.,'calibration_changes_trajectories':False,'policy_training':False,'test_evaluated':False}
    save(a.out/'comparison.json',result)
    assert all(sha(Path(p))==v for p,v in inputs.items())
    assert sha(a.out/'protocol.json')==manifest['protocol_sha256']
    assert sha(a.out/'operating_points.json')==manifest['operating_sha256_before_validation']
    assert sha(a.out/'calibrators.pkl')==manifest['calibrators_sha256_before_validation']
    assert sha(a.out/'models.pkl')==model_hash
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,checks_passed=True)
    save(a.out/'manifest.json',manifest);save(a.out/'progress.json',{'stage':'complete','elapsed_seconds':manifest['elapsed_seconds']})
    print('DONE',manifest['elapsed_seconds'],json.dumps(screen),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
