"""기존 대리 모델이 실제 위반 시작과 지속시간을 놓치는 정도를 집계한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import subprocess

import numpy as np
import pandas as pd
from history_trial import arrays, simulate, Surrogate, write, event_metrics


def main(a):
    if a.out.exists():raise ValueError(f'Refusing overwrite {a.out}')
    data=pd.read_pickle(a.data)
    report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'test_used':False,'rows':[]}
    for split in ('train','val'):
        x=arrays(data,split)
        truth=np.stack((x['nh4'][:,1:],x['do'][:,1:]),-1)
        initial=np.stack((x['nh4'][:,0],x['do'][:,0]),-1)
        for label,path in [('original',a.original),('e09_multistep',a.candidate)]:
            sur=Surrogate(path)
            for member in range(sur.n_members):
                pred=simulate(sur,x,member)
                report['rows'].append({'split':split,'model':label,'member':member,
                    'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'days':len(truth),'steps_per_day':287,**event_metrics(pred,truth,initial)})
    report['completed_at']=datetime.now(timezone.utc).isoformat()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write(a.out,report)


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser()
    for key in ('data','original','candidate','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
