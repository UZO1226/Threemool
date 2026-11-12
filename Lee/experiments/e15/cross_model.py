"""Frozen final controllers on alternative validation simulators, never test."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE=Path(__file__).resolve().parents[2]/'code/darrow'
sys.path.insert(0,str(CODE))
import fcsrl_compat
from fcsrl.utils import DeviceConfig
from train_dw import make_agent
from ppo_config import PPOConfig
from surrogate_dw import GBMDynamics
from env_dw import batched_rollout


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    os.environ['DARROW_FAST_DYNAMICS']='1'
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    results={}
    for model in a.models:
        dyn=GBMDynamics(model)
        rows=[]
        for method in ['ppolag','ppolag_fcsrl','ppolag_fcsrl_curriculum']:
            for seed in range(5):
                path=a.runs/method/f'seed_{seed}'
                config=PPOConfig.from_dict(json.loads((path/'model_final/config.json').read_text()))
                args=SimpleNamespace(method=method,fpen_weight=.1)
                agent,_=make_agent(args,config)
                agent.load(path/'model_final')
                with torch.no_grad():
                    score,_=batched_rollout(a.data,dyn,'val',lambda x:agent.act_det(x)[:,0],config.cost_limit)
                rows.append({'method':method,'seed':seed,**score})
        summary={method:{key:float(np.mean([r[key] for r in rows if r['method']==method]))
            for key in ['air_m3','cost','nh4_viol_pct','do_viol_pct']}
            for method in ['ppolag','ppolag_fcsrl','ppolag_fcsrl_curriculum']}
        results[str(model)]={'per_seed':rows,'means':summary}
        print(model,json.dumps(summary),flush=True)
    a.out.write_text(json.dumps({'split':'val','retuned':False,'models':results},indent=2,allow_nan=False))


if __name__=='__main__':
    main()
