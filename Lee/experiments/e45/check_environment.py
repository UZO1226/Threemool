"""Verify auxiliary-only adapter equivalence through a real daily reset."""
import argparse
import json
from pathlib import Path

import numpy as np

from severity_agent import SeverityInfoEnv,severity_cost
from quantile_env import FrozenDynamics,FrozenQuantileEnv

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();dynamics=FrozenDynamics(a.freeze)
base=FrozenQuantileEnv(a.data,dynamics,n_env=4,seed=45)
candidate=SeverityInfoEnv(a.data,dynamics,n_env=4,seed=45)
np.testing.assert_array_equal(base.reset(),candidate.reset())
base.set_member(2);candidate.set_member(2)
maximum=0.
for t in range(95):
    action=np.sin(np.arange(4)+t*.17)
    left=base.step(action);right=candidate.step(action)
    for i in range(5):np.testing.assert_array_equal(left[i],right[i])
    for key in left[5]:np.testing.assert_array_equal(left[5][key],right[5][key])
    np.testing.assert_array_equal(right[5]['next_state_for_aux'],right[4][:,:5])
    labels=severity_cost(right[2],right[5]['next_state_for_aux'])
    np.testing.assert_array_equal(labels==0,right[2]==0)
    maximum=max(maximum,float(labels.max()))
    assert right[3].all()==(t==94)
    np.testing.assert_array_equal(base.u,candidate.u)
    np.testing.assert_array_equal(base.state,candidate.state)
out={'training_cases_only':True,'transitions':380,'all_original_outputs_exact':True,
    'noise_and_post_reset_state_exact':True,'outgoing_terminal_labels_exact':True,
    'same_zero_cost_set':True,'maximum_auxiliary_observed':maximum,
    'policy_training':False,'test_evaluated':False,'policy_performance_tested':False}
a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
