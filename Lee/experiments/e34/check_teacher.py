"""Check alignment and actual teacher-forced gradient with fixed context."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from joint_teacher import TeacherTransitions,PersistentContextModel,sequences
from latent import Data

p=argparse.ArgumentParser();p.add_argument('--data',required=True)
p.add_argument('--reference-summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
with open(a.reference_summary) as f:ref=json.load(f)
data=Data(pd.read_pickle(a.data),ref['fit_days']);days=ref['fit_days'][:2]
model=PersistentContextModel(data.sd,data.mean,data.mean_air,data.hi)
teacher=TeacherTransitions(data,days);idx=torch.tensor([0,1])
s,c,e,air,target=sequences(data,days);prediction=model(s,c,e,air)
local=teacher.predictions(model,idx)
np.testing.assert_allclose(prediction[:,0].detach().numpy(),local[:,0].detach().numpy(),rtol=1e-5,atol=1e-5)
np.testing.assert_array_equal(target.numpy(),teacher.target.numpy())
np.testing.assert_array_equal(c.numpy(),teacher.context.numpy())
later=float((prediction[:,1:]-local[:,1:]).abs().max().detach());assert later>1e-3
gradient=torch.autograd.grad(teacher(model,idx),model.episode_encoder[-1].weight)[0]
assert torch.isfinite(gradient).all() and float(gradient.abs().sum())>1e-6
result={'step0_alignment':True,'targets_and_fixed_context_match':True,
    'later_prediction_difference':later,'finite_nonzero_teacher_gradient':True,'test_evaluated':False}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
