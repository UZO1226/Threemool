"""내부 기준을 통과한 고정 모델을 기존 val에서 한 번 재생한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from routing_trial import routing_frame, replay, finalize, Model, arrays, event_metrics, write


def main(a):
    torch.set_num_threads(1)
    if a.out.exists():raise ValueError(f'Refusing overwrite {a.out}')
    prior=json.loads((a.trial/'results.json').read_text())
    assert prior['gate']['passed']
    data=pd.read_pickle(a.data)
    x=arrays(data,'val')
    frame=routing_frame(a.raw,end='2024-02-29')
    aux=np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in data['days']['val']])
    assert np.isfinite(aux).all() and len(aux)==43
    truth=np.stack((x['nh4'][:,13:],x['do'][:,13:]),-1)
    initial=np.stack((x['nh4'][:,12],x['do'][:,12]),-1)
    r={'started_at':datetime.now(timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.raw,a.data,a.trial/'results.json')},
       'config':prior['config'],'partition':{'val':data['days']['val']},
       'training':prior['training'],'training_performed':False,'test_used':False,'validation_reused_exploratory':True,
       'routing_known_fraction':aux[:,:,-2:].mean((0,1)).tolist(),'evaluation':[]}
    for saved in prior['training']:
        mode,seed=saved['mode'],saved['seed']
        path=a.trial/f'{mode}_s{seed}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==saved['model_sha256']
        model=Model([torch.zeros(2,n) if n!=8 else torch.zeros(2,4,2) for n in (2,6,8,2)])
        model.load_state_dict(torch.load(path,weights_only=True))
        for h in (1,12,275):
            pred=replay(model,x,aux,mode,h)
            r['evaluation'].append({'mode':mode,'seed':seed,'split':'val','reset_steps':h,'days':43,
                'evaluated_steps_per_day':275,**event_metrics(pred,truth,initial)})
    finalize(r,split='val')
    r['checks']={'saved_model_sha256_unchanged':True,'finite_predictions':True,'fixed_settings':True}
    r['completed_at']=datetime.now(timezone.utc).isoformat()
    write(a.out,r)
    print(json.dumps(r['gate']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('raw','data','trial','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
