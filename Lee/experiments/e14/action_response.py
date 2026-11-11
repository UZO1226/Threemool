"""Recursive response to fixed recorded-air multipliers under action limits."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/darrow'))
from env_dw import VecAerationEnv,episodes,AIR_MAX
from surrogate_dw import GBMDynamics

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--models',type=Path,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
os.environ['DARROW_FAST_DYNAMICS']='1'
result={}
for path in a.models:
    dyn=GBMDynamics(path)
    template=VecAerationEnv(a.data,None,split='val',n_env=1,dynamics=dyn)
    eps=episodes(template)
    scores={}
    for scale in [.5,.75,1,1.25,1.5]:
        env=VecAerationEnv(a.data,None,split='val',n_env=len(eps),dynamics=dyn)
        env.fixed=eps
        env.reset()
        states=[]
        costs=[]
        for t in range(95):
            air=np.array([env.day_arrays[env.day[i]]['air'][env.t[i]] for i in range(env.n)])*scale
            _,_,cost,_,terminal,_=env.step(np.clip(air/AIR_MAX*2-1,-1,1))
            states.append(terminal[:,:5])
            costs.append(cost)
        state=np.stack(states)
        scores[str(scale)]={'nh4':float(state[:,:,0].mean()),'do3':float(state[:,:,4].mean()),
            'cost':float(np.stack(costs).sum(0).mean())}
    result[str(path)]=scores
a.out.write_text(json.dumps({'split':'val','action_clipped_to_9000':True,'results':result},indent=2))
print(a.out.read_text(),flush=True)
