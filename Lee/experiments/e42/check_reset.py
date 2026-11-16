"""Reproduce the historical terminal reward alias and verify its isolation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantile_env import TerminalSafeOriginalEnv,VecAerationEnv

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()


class CopyDynamics:
    n_members=1
    def step(self,state,exo,zone,previous_zone,members):return state.copy()


raw=pd.read_pickle(a.data);days=raw['days']['train'][:2]
old=VecAerationEnv(a.data,None,n_env=1,dynamics=CopyDynamics(),seed=0)
fixed=TerminalSafeOriginalEnv(a.data,CopyDynamics(),n_env=1,seed=0)
for env in [old,fixed]:env.fixed=[(d,0) for d in days];env.reset()
for step in range(95):
    left=old.step(np.zeros(1));right=fixed.step(np.zeros(1))
    np.testing.assert_array_equal(left[0],right[0]);np.testing.assert_array_equal(left[2],right[2])
    np.testing.assert_array_equal(left[4],right[4])
    np.testing.assert_array_equal(right[1],[-.5]);np.testing.assert_array_equal(right[5]['air'],[4500])
    if step<94:
        np.testing.assert_array_equal(left[1],right[1]);np.testing.assert_array_equal(left[5]['air'],right[5]['air'])
assert abs(left[1][0]+.5)>1e-6
assert left[5]['air'][0]==raw['data'].loc[days[1]].air.iloc[0]
result={'historical_alias_reproduced':True,'legacy_terminal_reward':float(left[1][0]),
    'legacy_terminal_reported_air':float(left[5]['air'][0]),'actual_executed_air':4500,
    'corrected_terminal_reward':float(right[1][0]),'all95_corrected_rewards_pass':True,
    'state_cost_terminal_observations_unchanged':True,'test_evaluated':False}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
