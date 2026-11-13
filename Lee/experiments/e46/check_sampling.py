"""Check train-only weights, actual reset sampling and unchanged transitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cold_sampling import ColdTrainingEnv,cold_day_weights
from quantile_env import FrozenDynamics,FrozenQuantileEnv

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
days=['2021-01-01','2021-01-02','2021-01-03','2021-01-04']
table=pd.DataFrame({'temp':np.repeat([4.,1.,1.,3.],2)},
    index=pd.to_datetime([f'{day} {hour:02d}:00' for day in days for hour in (0,1)]))
weights,info=cold_day_weights(table,days)
np.testing.assert_allclose(weights,[1/6,.5,1/6,1/6]);assert info['cold_days']==days[1:2]
try:cold_day_weights(table,days[:1])
except ValueError:pass
else:raise AssertionError('Single-day input accepted')
table.loc[days[0],'temp']=np.nan
try:cold_day_weights(table,days)
except ValueError:pass
else:raise AssertionError('Nonfinite temperature accepted')

dyn=FrozenDynamics(a.freeze);base=FrozenQuantileEnv(a.data,dyn,n_env=4,seed=46)
candidate=ColdTrainingEnv(a.data,dyn,n_env=4,seed=46)
assert set(candidate.sampling_info['days'])==set(base.days) and len(base.days)==188
assert len(candidate.sampling_info['cold_days'])==47
assert np.isclose(candidate.day_weights.sum(),1)
assert min(candidate.day_weights)>0
cases=[(day,i) for i,day in enumerate(base.days[:4])]
base.fixed=cases;candidate.fixed=cases
np.testing.assert_array_equal(base.reset(),candidate.reset())
for t in range(95):
    action=np.cos(np.arange(4)+t*.11);left=base.step(action);right=candidate.step(action)
    for i in range(5):np.testing.assert_array_equal(left[i],right[i])
    for key in left[5]:np.testing.assert_array_equal(left[5][key],right[5][key])
    np.testing.assert_array_equal(base.u,candidate.u)
candidate.fixed=None;candidate.fix_member=0
cold=set(candidate.sampling_info['cold_days']);count=0
for _ in range(1024):
    candidate._pick(0);count+=candidate.day[0] in cold
fraction=count/1024;assert .44<fraction<.56
for split in ('val','test'):
    try:ColdTrainingEnv(a.data,dyn,split=split)
    except ValueError:pass
    else:raise AssertionError('Non-training sampler accepted')
out={'training_days_only':True,'stable_ties_and_weight_mass_pass':True,'positive_support_all_training_days':True,
    'invalid_input_rejected':True,'real_environment_step_outputs_exact':True,'compared_transitions':380,
    'actual_reset_draws':1024,'observed_cold_fraction':fraction,'expected_cold_fraction':.5,
    'validation_test_sampler_rejected':True,'test_evaluated':False,'policy_performance_tested':False,
    'sampling':candidate.sampling_info}
a.out.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='sampling'}),flush=True)
