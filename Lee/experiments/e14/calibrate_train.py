"""Calibrate from training replay only, without evaluating held-out dates."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/darrow'))
from env_dw import batched_rollout
from surrogate_dw import GBMDynamics

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--surrogate',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
s,_=batched_rollout(a.data,GBMDynamics(a.surrogate),'train',None,1e9,replay=True)
report={'train':s,'cost_limit':float(round(s['cost'])),'splits_evaluated':['train']}
a.out.write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
