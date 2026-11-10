"""저장 예측으로 MAE와 분류 수치를 독립 집계한다."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from history import STATE,initial_history,advance_history


def main(out):
    c=json.loads((out/'comparison.json').read_text());z=np.load(out/'predictions.npz')
    count=0
    for name in ('base','history'):
        for mode,t in [('cold',95),('warm',91)]:
            pred=z[name+'_'+mode];actual=z['actual_'+mode]
            assert pred.shape==(40,5,t,5) and actual.shape==(40,t,5)
            for j,target in enumerate(STATE):
                assert np.isclose(np.abs(pred[:,:,:,j]-actual[:,None,:,j]).mean(),c['results'][name][mode]['mae'][target])
                count+=1
            for target in ('NH4','DO'):
                a=actual[:,:,0]>4 if target=='NH4' else actual[:,:,2:].max(2)>2
                p=pred[:,:,:,0]>4 if target=='NH4' else pred[:,:,:,2:].max(3)>2
                a=np.repeat(a[:,None],5,axis=1)
                tp=int((a&p).sum());fp=int((~a&p).sum());cf=c['results'][name][mode]['confusion'][target]
                assert tp==cf['tp'] and fp==cf['fp']
                assert np.isclose(100*tp/a.sum(),cf['recall_pct'])
                assert np.isclose(100*fp/(~a).sum(),cf['fpr_pct'])
                count+=1
    for name in ('base','history'):
        for group,sign in [('increase',1),('decrease',-1)]:
            mask=z['event_dq']*sign>0
            for q in c['results'][name]['events'][group]['metrics']:
                j=STATE.index(q['target']);t=q['minutes']//15
                pred=z[name+'_events'][mask,:,t,j].mean(1);actual=z['actual_events'][mask,t,j]
                assert np.isclose(np.abs(pred-actual).mean(),q['mae'])
                count+=1
    s=np.arange(50).reshape(10,5);h=initial_history(s,4)[None]
    assert np.array_equal(h[0],s[[3,2,1,0]])
    assert np.array_equal(initial_history(s,1),np.repeat(s[:1],4,axis=0))
    assert np.array_equal(advance_history(h,s[4:5])[0],s[[4,3,2,1]])
    m=json.loads((out/'manifest.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert all(sha(p)==v for p,v in m['input_sha256'].items())
    assert sha(out/'candidate.pkl')==m['candidate_sha256']
    assert sha(out/'protocol.json')==m['protocol_sha256']
    result={'checks_passed':True,'independent_aggregate_checks':count,'history_order_padding_shift_pass':True,'input_protocol_candidate_hashes_pass':True,'script_sha256':sha(__file__)}
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
