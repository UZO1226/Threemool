"""사후 진단: 위험 순위와 고정 농도 경계에서의 성능을 구분한다."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score


def main(out):
    z=np.load(out/'predictions.npz')
    truth=np.repeat(z['actual_cold'][:,None,:,0]>4,5,axis=1)
    scores={key:z[key+'_cold'][:,:,:,0] for key in ('base','history')}
    at4={key:{'recall':float((s[truth]>4).mean()),'fpr':float((s[~truth]>4).mean())} for key,s in scores.items()}
    result={'posthoc_diagnostic_only':True,'truth_threshold_mg_L':4.,
            'no_prediction_threshold_deployed':True,'original_screen_unchanged':True,
            'note':'Thresholds chosen on development validation outcomes describe ranking potential only. They are not trained calibration, independent performance estimates, or a change to the physical threshold.', 'models':{}}
    y=truth.ravel();target=at4['base']['recall']
    for key,s in scores.items():
        fpr,tpr,threshold=roc_curve(y,s.ravel(),drop_intermediate=False)
        row={'auc':float(roc_auc_score(y,s.ravel())),'average_precision':float(average_precision_score(y,s.ravel())),
             'at_prediction_threshold_4':at4[key],'matched_recall':[],'matched_fpr':[]}
        for recall in (target,.8):
            i=int(np.flatnonzero(tpr>=recall)[0])
            row['matched_recall'].append({'target_recall':recall,'achieved_recall':float(tpr[i]),'fpr':float(fpr[i]),'exploratory_score_threshold':float(threshold[i])})
        for cap in (.1,at4['base']['fpr']):
            i=int(np.flatnonzero(fpr<=cap)[-1])
            row['matched_fpr'].append({'fpr_cap':cap,'achieved_fpr':float(fpr[i]),'recall':float(tpr[i])})
        result['models'][key]=row
    rng=np.random.default_rng(3801);starts=rng.integers(40,size=(2000,6))
    ids=((starts[:,:,None]+np.arange(7))%40).reshape(2000,-1)[:,:40]
    diffs=[]
    for ii in ids:
        yy=truth[ii].ravel()
        diffs.append(roc_auc_score(yy,scores['history'][ii].ravel())-roc_auc_score(yy,scores['base'][ii].ravel()))
    result['history_minus_base_auc_paired_7day_block_95ci']=np.quantile(diffs,[.025,.975]).tolist()
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'ranking.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
