"""Regenerate the excluded inner PIT bank from authorized data and checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantile_env import WORKSPACE
from quantile_dynamics import QuantileDynamics,build_arrays,pit

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
freeze=json.loads(a.freeze.read_text());runs=WORKSPACE/freeze['environment_run']
expected_data=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected_data
for name,digest in freeze['frozen_files_sha256'].items():
    path=WORKSPACE/name
    if path.suffix=='.pkl':assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
summary=json.loads((runs/'summary.json').read_text());raw=pd.read_pickle(a.data)
import os
os.environ['DARROW_FAST_DYNAMICS']='1'
dyn=QuantileDynamics(runs);x,delta=build_arrays(raw['data'],summary['calibration_days'])
members=np.repeat(np.arange(5),len(x));values=dyn.quantiles(np.tile(x,(5,1)),members)
target=np.tile(x[:,:5]+delta,(5,1))
bank=pit(values,target,np.random.default_rng(3801)).reshape(5,len(x),5);bank[...,1]=.5
np.savez_compressed(a.out,pit=bank)
expected=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('pit_bank.npz'))
actual=hashlib.sha256(a.out.read_bytes()).hexdigest();assert actual==expected
print(json.dumps({'sha256':actual,'matches_frozen_bank':True}),flush=True)
