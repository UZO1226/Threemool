"""주 매칭 결과를 바꾸지 않고 거리 기준의 보수성과 기록 행동을 점검한다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e30'))
from darrow_env import load_days


def main(a):
    p=json.loads((a.out/'protocol.json').read_text())
    result=json.loads((a.out/'results.json').read_text())
    z=np.load(a.out/'trajectories.npz')
    n=result['support']['val_events']
    x=(z['context']-np.array(p['center']))/np.array(p['iqr_scale'])
    count={str(v):0 for v in (1.,2.,3.)}
    nearest_failures={k:0 for k in p['features']}
    matches=json.loads((a.out/'matches.json').read_text())
    for m in matches:
        d=np.abs(x[m['event_index']]-x[n+m['control_index']])
        for j,key in enumerate(p['features']):
            nearest_failures[key]+=int(d[j]>1)
    for i in range(n):
        diff=np.abs(x[n:]-x[i])
        rms=np.sqrt((diff**2).mean(1))
        cross=z['days'][n:]!=z['days'][i]
        for cutoff in (1.,2.,3.):
            count[str(cutoff)]+=int(np.any(cross & (rms<=p['rms_distance_limit']) & (diff.max(1)<=cutoff)))
    _,days=load_days(a.prepared/'data.pkl')
    groups={}
    for name,direction in [('increase',1),('decrease',-1)]:
        ii=np.flatnonzero(z['dq'][:n]*direction>0)
        net=[]
        for i in ii:
            day=str(z['days'][i]);t=int(z['t'][i]);g=days[day]
            net.append([float((g['air'][t:t+h].mean()-g['air'][t-1])*direction) for h in (1,4,12)])
        net=np.array(net)
        groups[name]={'events':len(ii),'initial_nh4_mean':float(z['actual'][ii,0,0].mean()),
                      'previous_hour_nh4_change_mean':float((z['context'][ii,5]*4).mean()),
                      'signed_average_air_change_vs_previous_at_15_60_180min':net.mean(0).tolist(),
                      'same_direction_average_exposure_pct':(100*(net>0).mean(0)).tolist()}
    out={'post_primary_descriptive_sensitivity':True,
         'primary_nearest_then_filter_pairs':result['support']['accepted_pairs'],
         'events_with_ANY_eligible_crossday_control_by_max_iqr':count,
         'nearest_pair_feature_fail_counts_over_1iqr':nearest_failures,
         'observed_action_context':groups,
         'primary_results_changed':False,
         'warning':'Cutoffs 2 and 3 are post-primary diagnostics, not predeclared success thresholds or causal identification.',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'support_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
