"""Check physical-step alignment and actual gradient in local supervision."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from joint_horizon import LocalTransitions,sequences
from latent import Data,LatentModel

p=argparse.ArgumentParser();p.add_argument('--data',required=True)
p.add_argument('--reference-summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
raw=pd.read_pickle(a.data)
with open(a.reference_summary) as f:ref=json.load(f)
data=Data(raw,ref['fit_days']);days=ref['fit_days'][:2]
model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
local=LocalTransitions(data,days);idx=torch.tensor([0,1])
s,c,e,air,target=sequences(data,days)
full=model(s,c,e,air);one=local.predictions(model,idx)
np.testing.assert_allclose(one[:,0].detach().numpy(),full[:,0].detach().numpy(),rtol=1e-4,atol=1e-4)
np.testing.assert_array_equal(local.target.numpy(),target.numpy())
later=float((one[:,1:]-full[:,1:]).abs().max().detach());assert later>1e-3
loss=local(model,idx);gradient=torch.autograd.grad(loss,model.forcing[-1].bias)[0]
assert torch.isfinite(gradient).all() and float(gradient[:5].abs().sum())>1e-6
result={'step0_alignment':True,'target_alignment':True,'later_prediction_difference':later,
    'finite_nonzero_local_loss_gradient':True,'test_evaluated':False}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
