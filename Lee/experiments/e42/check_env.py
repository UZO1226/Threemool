"""Verify the frozen control adapter against the validated forecast runner."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantile_env import FrozenDynamics,FrozenQuantileEnv,AIR_MAX,ZONES
from quantile_dynamics import arrays,paths,rollout

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();dyn=FrozenDynamics(a.freeze);raw=pd.read_pickle(a.data)
days=raw['days']['val'][:2];draws=2
cases=[(d,m,r) for m in range(5) for r in range(draws) for d in days]
batch=arrays(raw['data'],days);noise=paths(dyn.bank,days,95,draws=draws)
expected=rollout(dyn.model,batch,noise,.5,draws=draws).reshape(len(cases),95,5)
env=FrozenQuantileEnv(a.data,dyn,split='val',n_env=len(cases),seed=0)
env.fixed=[(d,m) for d,m,r in cases];env.fixed_draws=[r for d,m,r in cases]
obs=env.reset();np.testing.assert_array_equal(env.u,noise)
expected_fractions=np.median(np.concatenate([raw['data'].loc[d][ZONES].to_numpy(float) for d in raw['days']['train']]),axis=0)
expected_fractions/=expected_fractions.sum();np.testing.assert_array_equal(env.fractions,expected_fractions)
trajectory=[];total_cost=np.zeros(env.n);volume=np.zeros(env.n)
for t in range(95):
    recorded=np.array([env.day_arrays[d]['air'][t] for d,m,r in cases])
    obs,reward,cost,done,terminal,info=env.step(recorded/AIR_MAX*2-1)
    trajectory.append(terminal[:,:5].copy())
    np.testing.assert_allclose(reward,-recorded/AIR_MAX,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(cost,(expected[:,t,0]>4).astype(float)+(expected[:,t,2:].max(-1)>2).astype(float))
    assert done.all()==(t==94);assert obs.shape==(len(cases),11)
    total_cost+=cost;volume+=info['air']*.25
np.testing.assert_allclose(np.stack(trajectory,axis=1),expected,rtol=2e-6,atol=2e-6)
assert np.all(env.t==0)
expected_cost=(expected[...,0]>4).sum(1)+(expected[...,2:].max(-1)>2).sum(1)
np.testing.assert_array_equal(total_cost,expected_cost)
np.testing.assert_allclose(volume,np.tile(np.array([raw['data'].loc[d].air.iloc[:95].sum()*.25 for d in days]),5*draws))

left=FrozenQuantileEnv(a.data,dyn,n_env=2,seed=41);right=FrozenQuantileEnv(a.data,dyn,n_env=2,seed=41)
np.testing.assert_array_equal(left.reset(),right.reset());left.set_member(3);right.set_member(3)
np.testing.assert_array_equal(left.u,right.u);assert left.day==right.day
low=rollout(dyn.model,batch,noise,.5,draws=draws,air_scale=.75)
high=rollout(dyn.model,batch,noise,.5,draws=draws,air_scale=1.25)
assert np.maximum(high[...,0]-low[...,0],0).max()<1e-8
assert np.maximum(low[...,2:]-high[...,2:],0).max()<1e-8
try:FrozenQuantileEnv(a.data,dyn,split='test')
except ValueError:pass
else:raise AssertionError('Reserved split was accepted')
result={'frozen_hashes_verified':True,'full_day_forecast_adapter_agreement':True,
    'validation_paths_exact':True,'training_noise_reproducible':True,
    'fallback_training_medians_exact':True,'reward_cost_volume_exact':True,
    'terminal_observation_before_reset':True,'common_noise_air_order':True,
    'test_split_rejected':True,'test_evaluated':False,'episodes_checked':len(cases),
    'maximum_forecast_difference':float(np.abs(np.stack(trajectory,axis=1)-expected).max())}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
