"""Check whether a fixed baseline action bias explains the extension trade-off."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/darrow'))
import fcsrl_compat
from fcsrl.utils import DeviceConfig
from train_dw import make_agent
from ppo_config import PPOConfig
from surrogate_dw import GBMDynamics
from env_dw import batched_rollout

p=argparse.ArgumentParser()
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
os.environ['DARROW_FAST_DYNAMICS']='1'
torch.set_num_threads(1)
DeviceConfig().select_device(-1)
dyn=GBMDynamics(a.model)
rows=[]
for seed in range(5):
    run=a.runs/'ppolag_fcsrl'/f'seed_{seed}'
    config=PPOConfig.from_dict(json.loads((run/'model_final/config.json').read_text()))
    agent,_=make_agent(SimpleNamespace(method='ppolag_fcsrl'),config)
    agent.load(run/'model_final')
    for bias in [-.2,-.1,0,.1,.2]:
        score,_=batched_rollout(a.data,dyn,'val',lambda x:np.clip(agent.act_det(x)[:,0]+bias,-1,1),config.cost_limit)
        if bias==0:
            saved=json.loads((run/'metrics.json').read_text())['val_final']
            for key in ['air_m3','cost','nh4_viol_pct','do_viol_pct']:
                np.testing.assert_allclose(score[key],saved[key],rtol=1e-7,atol=1e-7)
        rows.append({'seed':seed,'normalized_action_bias':bias,**score})
    print('Finished baseline air sweep seed',seed,flush=True)
means=[{'bias':bias,**{key:float(np.mean([r[key] for r in rows if r['normalized_action_bias']==bias]))
        for key in ['air_m3','cost','nh4_viol_pct','do_viol_pct']}} for bias in [-.2,-.1,0,.1,.2]]
ext=[json.loads((a.runs/'ppolag_fcsrl_curriculum'/f'seed_{seed}'/'metrics.json').read_text())['val_final'] for seed in range(5)]
air=float(np.mean([r['air_m3'] for r in ext]))
cost=float(np.mean([r['cost'] for r in ext]))
curve=sorted(means,key=lambda row:row['air_m3'])
interpolated=float(np.interp(air,[r['air_m3'] for r in curve],[r['cost'] for r in curve])) if curve[0]['air_m3']<=air<=curve[-1]['air_m3'] else None
result={'split':'val','baseline_retrained':False,'zero_bias_reproduces_saved_final':True,
    'caveat':'Fixed-action-bias sensitivity, not a retrained Pareto frontier. No test evaluation.',
    'rows':rows,'means':means,'extension_mean_air':air,'extension_mean_cost':cost,
    'baseline_interpolated_cost_at_extension_air':interpolated,
    'extension_cost_minus_interpolated_baseline':None if interpolated is None else cost-interpolated}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','means']}),flush=True)
