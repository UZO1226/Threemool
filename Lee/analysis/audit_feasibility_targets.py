"""Audit learned feasibility and day-end bootstraps on training rollouts only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e44'))
from evaluate_pair import load_run,file_hashes,FrozenDynamics
from quantile_env import FrozenQuantileEnv
from ppo_fcsrl import feasibility_targets
from fcsrl.utils import DiscDist

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
torch.set_num_threads(1);dyn=FrozenDynamics(a.freeze)
days=pd.read_pickle(a.data)['days']['train'];selected=[days[i] for i in np.linspace(0,len(days)-1,16,dtype=int)]
cases=[(day,member,draw) for member in range(5) for draw in range(2) for day in selected]
result={'training_only':True,'validation_evaluated':False,'test_evaluated':False,'policy_refit':False,
    'training_dates':len(selected),'cases_per_seed':len(cases),'transitions_per_case':95,
    'scope':'Descriptive fixed-policy training-rollout audit. Scores predict discounted maxima, not probabilities. Finite-day targets differ from the existing continuation-bootstrap objective. No actor or representation is updated.',
    'policies':[]}
for seed in range(5):
    folder=a.baseline_root/f'seed_{seed}';agent,config,metrics,hashes=load_run(folder,a.freeze)
    assert config['method']=='ppolag_fcsrl'
    env=FrozenQuantileEnv(a.data,dyn,split='train',n_env=len(cases),seed=0)
    env.fixed=[(day,member) for day,member,draw in cases];env.fixed_draws=[draw for day,member,draw in cases]
    obs=env.reset();cost=[];q=[];next_f=[]
    for step in range(95):
        action=np.clip(agent.act_det(obs),-1,1)
        with torch.no_grad():
            z=agent.encoder.zs(agent._t(agent.obs_normalizer(obs)))
            predicted=agent.encoder.zsa(z,agent._t(action))
            logits=agent.feasi_head(predicted).unsqueeze(0)
            q.append(DiscDist(logits,agent.bucket_low,agent.bucket_high,agent.n_buckets).mean().cpu().numpy().reshape(-1))
        obs,reward,c,done,terminal,info=env.step(action[:,0]);cost.append(c)
        next_f.append(agent.next_feasibility(agent.obs_normalizer(terminal)))
        assert done.all()==(step==94)
    cost=np.asarray(cost,dtype=np.float32);q=np.asarray(q);next_f=np.asarray(next_f)
    end=np.zeros_like(cost);end[-1]=1
    continuation=feasibility_targets(cost,next_f,end,np.zeros_like(cost),.9)
    finite=feasibility_targets(cost,next_f,end,end,.9)
    delta=continuation-finite;assert delta.min()>-1e-6
    assert np.isfinite(q).all() and np.isfinite(continuation).all()
    ranks_q=pd.Series(q.ravel()).rank();ranks_target=pd.Series(continuation.ravel()).rank()
    correlation=float(ranks_q.corr(ranks_target)) if q.std()>1e-10 and continuation.std()>1e-10 else None
    order=np.argsort(q.ravel(),kind='stable');bins=[]
    for indices in np.array_split(order,4):
        bins.append({'predicted_mean':float(q.ravel()[indices].mean()),
            'continuation_target_mean':float(continuation.ravel()[indices].mean()),
            'immediate_cost_mean':float(cost.ravel()[indices].mean())})
    row={'seed':seed,'score_std':float(q.std()),'target_std':float(continuation.std()),
        'score_target_mse':float(((q-continuation)**2).mean()),'score_target_rank_correlation':correlation,
        'score_ordered_quartiles':bins,'continuation_minus_finite_mean':float(delta.mean()),
        'continuation_minus_finite_max':float(delta.max()),
        'fraction_targets_differ_above_0_001':float((delta>1e-3).mean()),
        'first71_steps_mean_difference':float(delta[:71].mean()),'last24_steps_mean_difference':float(delta[71:].mean()),
        'positive_day_end_bootstrap_fraction':float((next_f[-1]>0).mean()),
        'checkpoint_unchanged':file_hashes(folder/'model_final')==hashes}
    assert row['checkpoint_unchanged'];result['policies'].append(row)
    print('AUDIT_COMPLETE',seed,flush=True)
a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
