"""Verify optimized dynamics against sklearn on held-out validation inputs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/darrow'))
from surrogate_dw import GBMDynamics,build_arrays
from curriculum_dw import risk_day_weights

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--models',type=Path,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
d=pd.read_pickle(a.data)
x,_=build_arrays(d['data'],d['days']['val'])
reports=[]
for path in a.models:
    os.environ['DARROW_FAST_DYNAMICS']='0'
    slow=GBMDynamics(path)
    os.environ['DARROW_FAST_DYNAMICS']='1'
    fast=GBMDynamics(path)
    errors=[]
    for m in range(5):
        member=np.full(len(x),m)
        expected=slow.delta(x,member)
        actual=fast.delta(x,member)
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)
        errors.append(float(np.max(np.abs(actual-expected))))
    member=np.arange(len(x))%5
    np.testing.assert_allclose(fast.delta(x,member),slow.delta(x,member),rtol=1e-11,atol=1e-11)
    timings={}
    for name,model in [('sklearn',slow),('fast',fast)]:
        start=time.perf_counter()
        for _ in range(100): model.delta(x[:8],np.zeros(8,dtype=int))
        timings[name]=time.perf_counter()-start
    reports.append({'model':str(path),'max_absolute_error':max(errors),'rows_per_member':len(x),
                    'members':5,'mixed_members_checked':True,'100_small_batch_seconds':timings})
weights,info=risk_day_weights(d['data'],d['days']['train'])
assert np.isclose(weights.sum(),1)
assert set(info['high_risk_days']).issubset(d['days']['train'])
assert np.isclose(sum(weights[d['days']['train'].index(day)] for day in info['high_risk_days']),.5)
a.out.write_text(json.dumps({'equivalence':reports,'curriculum_checks':'passed'},indent=2))
print(a.out.read_text(),flush=True)
