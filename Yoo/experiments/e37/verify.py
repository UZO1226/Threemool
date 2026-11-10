"""저장 궤적의 정렬, MAE, 매칭과 입력 불변을 독립 검산한다."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run import context, classes, events
from darrow_env import load_days
from common import sha


def main(a):
    z=np.load(a.out/'trajectories.npz')
    r=json.loads((a.out/'results.json').read_text())
    p=json.loads((a.out/'protocol.json').read_text())
    m=json.loads((a.out/'manifest.json').read_text())
    _,days=load_days(a.prepared/'data.pkl')
    n=r['support']['val_events'];actual=z['actual'];pred=z['pred']
    assert pred.shape==(len(actual),5,13,5)
    assert np.allclose(pred[:,:,0],actual[:,None,0])
    assert np.all(np.isfinite(pred))
    for i in range(len(actual)):
        d=str(z['days'][i]);t=int(z['t'][i]);g=days[d]
        assert np.array_equal(actual[i],g['state'][t:t+13])
        assert np.array_equal(z['context'][i],context(g,t))
        assert np.isclose(z['dq'][i],g['air'][t]-g['air'][t-1])
    for d in np.unique(z['days'][:n]):
        ts=z['t'][:n][z['days'][:n]==d]
        assert np.all(np.diff(ts)>=12)
    for group in ('all_changes','increase','decrease'):
        mask=np.ones(n,bool) if group=='all_changes' else z['dq'][:n]*(1 if group=='increase' else -1)>0
        assert int(mask.sum())==r['response'][group]['events']
        for q in r['response'][group]['metrics']:
            h=q['minutes']//15;j=['nh4','no3','do1','do2','do3'].index(q['target'])
            obs=actual[:n][mask,h,j];model=pred[:n][mask,:,h,j].mean(1)
            assert np.isclose(np.abs(model-obs).mean(),q['mae'])
            assert np.isclose(np.abs(obs-actual[:n][mask,0,j]).mean(),q['persistence_mae'])
    matches=json.loads((a.out/'matches.json').read_text())
    for q in matches:
        assert q['event_day']!=q['control_day']
        e=q['event_index'];c=n+q['control_index']
        diff=np.abs((z['context'][e]-z['context'][c])/np.array(p['iqr_scale']))
        assert np.isclose(np.sqrt((diff**2).mean()),q['rms_iqr_distance'])
        assert q['accepted']==bool(np.sqrt((diff**2).mean())<=p['rms_distance_limit'] and diff.max()<=1)
    assert all(sha(Path(f))==h for f,h in m['input_sha256'].items())
    assert sha(a.out/'protocol.json')==m['protocol_sha256']
    assert classes(np.array([-.11,-.1,0,.1,.11]),.1).tolist()==[-1,0,0,0,1]
    synthetic=[{'day':'a','t':t,'dq':600} for t in (4,5,16,17,28)]
    assert [x['t'] for x in events(synthetic,500)]==[4,16,28]
    out={'checks_passed':True,'windows_verified':len(actual),'events_verified':n,
         'aggregate_metric_rows_verified':45,'script_sha256':sha(Path(__file__))}
    (a.out/'verification.json').write_text(json.dumps(out,indent=2))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    main(p.parse_args())
