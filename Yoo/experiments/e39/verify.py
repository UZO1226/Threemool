"""날짜 분리, 고정 판정점과 분류 집계를 독립 확인한다."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def main(out):
    p=json.loads((out/'protocol.json').read_text());m=json.loads((out/'manifest.json').read_text());c=json.loads((out/'comparison.json').read_text());o=json.loads((out/'operating_points.json').read_text())
    fit,cal,val=[p[k] for k in ('fit_dates','calibration_dates','validation_dates')]
    assert (len(fit),len(cal),len(val))==(140,48,40)
    assert max(fit)<min(cal) and max(cal)<min(val)
    assert not(set(fit)&set(cal) or set(cal)&set(val) or set(fit)&set(val))
    with (out/'calibrators.pkl').open('rb') as f:iso=pickle.load(f)
    z=np.load(out/'predictions.npz');truth=np.repeat(z['actual'][:,None,:,0]>4,5,axis=1)
    ycal=np.repeat(z['actual_calibration'][:,None,:,0]>4,5,axis=1)
    count=0
    for name in ('base','history'):
        scores=z[name+'_pred'][:,:,:,0];prob=iso[name].predict(scores.ravel()).reshape(scores.shape)
        assert np.array_equal(prob,z[name+'_prob'])
        cp=iso[name].predict(z[name+'_calibration_pred'][:,:,:,0].ravel()).reshape(ycal.shape)
        for mode in ('raw','recall80','fpr10'):
            alerts=scores>4 if mode=='raw' else prob>=o['models'][name][mode]['probability_threshold']
            assert np.array_equal(alerts,z[name+'_alert_'+mode])
            x=c['validation'][name]['nh4'][mode]
            tp=int((truth&alerts).sum());fp=int((~truth&alerts).sum())
            assert tp==x['tp'] and fp==x['fp']
            assert np.isclose(100*tp/truth.sum(),x['recall_pct'])
            assert np.isclose(100*fp/(~truth).sum(),x['fpr_pct'])
            counts=np.array(x['day_counts']);assert counts[:,0].sum()==tp and counts[:,1].sum()==fp
            count+=1
            if mode!='raw':
                ca=cp>=o['models'][name][mode]['probability_threshold']
                recall=100*ca[ycal].mean();fpr=100*ca[~ycal].mean()
                assert np.isclose(recall,o['models'][name][mode]['calibration_recall_pct'])
                assert np.isclose(fpr,o['models'][name][mode]['calibration_fpr_pct'])
                assert recall>=80-1e-10 if mode=='recall80' else fpr<=10+1e-10
        # Label calibration has no path back into the saved recursive state trajectories.
        for j,key in enumerate(('nh4','no3','do1','do2','do3')):
            assert np.isclose(np.abs(z[name+'_pred'][:,:,:,j]-z['actual'][:,None,:,j]).mean(),c['validation'][name]['mae'][key])
            count+=1
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert all(sha(path)==v for path,v in m['input_sha256'].items())
    for filename,key in [('protocol.json','protocol_sha256'),('operating_points.json','operating_sha256_before_validation'),('calibrators.pkl','calibrators_sha256_before_validation'),('models.pkl','model_sha256')]:
        assert sha(out/filename)==m[key]
    result={'checks_passed':True,'independent_metric_checks':count,'disjoint_dates':True,'calibration_targets_and_frozen_hashes':True,'source_sha256':sha(__file__)}
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
