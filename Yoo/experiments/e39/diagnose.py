"""사후 기술 점검: 기간 차이와 단조 보정의 동점 묶음을 분리한다."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e30'))
from darrow_env import load_days


def main(a):
    p=json.loads((a.out/'protocol.json').read_text());z=np.load(a.out/'predictions.npz')
    _,days=load_days(a.prepared/'data.pkl')
    split={}
    for name,key in [('fit','fit_dates'),('calibration','calibration_dates'),('val','validation_dates')]:
        dates=p[key];states=np.stack([days[d]['state'][1:] for d in dates]);temps=np.array([days[d]['exo'][:,1].mean() for d in dates])
        split[name]={'dates':len(dates),'nh4_exceedance_pct':float((states[:,:,0]>4).mean()*100),
                     'days_with_any_nh4_exceedance':int((states[:,:,0]>4).any(1).sum()),
                     'mean_daily_temperature':float(temps.mean()),'min_daily_temperature':float(temps.min()),'max_daily_temperature':float(temps.max())}
    with (a.out/'calibrators.pkl').open('rb') as f:cal=pickle.load(f)
    y=np.repeat(z['actual_calibration'][:,None,:,0]>4,5,axis=1).ravel();details={}
    for name in ('base','history'):
        score=z[name+'_calibration_pred'][:,:,:,0].ravel();prob=cal[name].predict(score)
        highest=prob==prob.max();fpr,tpr,threshold=roc_curve(y,score,drop_intermediate=False)
        i=int(np.flatnonzero(fpr<=.1)[-1]);j=int(np.flatnonzero(tpr>=.8)[0])
        details[name]={'distinct_calibrated_levels':len(np.unique(prob)), 'maximum_calibrated_probability':float(prob.max()),
                       'highest_probability_group_pct':float(highest.mean()*100),
                       'fpr_if_alert_highest_group_pct':float(highest[~y].mean()*100),
                       'recall_if_alert_highest_group_pct':float(highest[y].mean()*100),
                       'calibration_raw_auc':float(roc_auc_score(y,score)),'calibration_isotonic_auc':float(roc_auc_score(y,prob)),
                       'raw_score_fpr10_calibration_recall_pct':float(tpr[i]*100),'raw_score_fpr10_calibration_fpr_pct':float(fpr[i]*100),
                       'raw_score_recall80_calibration_recall_pct':float(tpr[j]*100),'raw_score_recall80_calibration_fpr_pct':float(fpr[j]*100)}
    result={'posthoc_descriptive_only':True,'no_new_thresholds_applied_to_validation':True,'periods':split,'calibration_plateaus':details,
            'interpretation':'Quantization and period shifts are observed properties. They do not separately identify the causal contribution to validation failure or prove no useful classifier exists.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
