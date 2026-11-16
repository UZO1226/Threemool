"""Evaluate one complete final policy for the frozen-model cross comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_pair import load_run,FrozenDynamics,evaluate,WORKSPACE,frozen_digest,GBMDynamics,file_hashes

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
p.add_argument('--environment',choices=['old','new'],required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
torch.set_num_threads(1);os.environ['DARROW_FAST_DYNAMICS']='1'
freeze=json.loads(a.freeze.read_text())
expected=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected
agent,config,metrics,hashes=load_run(a.run,a.freeze)
original=WORKSPACE/freeze['original_environment_path']
assert hashlib.sha256(original.read_bytes()).hexdigest()==freeze['original_environment_sha256']
dynamics=FrozenDynamics(a.freeze) if a.environment=='new' else GBMDynamics(original)
summary,rows=evaluate(a.data,dynamics,act_batch=lambda x:agent.act_det(x)[:,0])
reproduced=config['experiment_metadata']['environment']==a.environment
if reproduced:
    for key,value in metrics['val_final'].items():np.testing.assert_allclose(summary[key],value,rtol=1e-10,atol=1e-10)
assert file_hashes(a.run/'model_final')==hashes
result={'policy_seed':config['seed'],'method':config['method'],'evaluation_environment':a.environment,
    'training_environment':config['experiment_metadata']['environment'],'split':'val','summary':summary,'rows':rows,
    'freeze_canonical_sha256':frozen_digest(a.freeze),'checkpoint_hashes':hashes,
    'training_steps':metrics['steps'],'stored_native_metrics_reproduced':reproduced,
    'checkpoint_loaded_and_evaluated':True,'test_evaluated':False,'smoke_only':False}
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
