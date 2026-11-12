"""Independent equilibrium and gradient check, with a deliberately broken control."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
from latent import Data,LatentModel

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
raw=pd.read_pickle(a.data);data=Data(raw,raw['days']['train'][:-30]);torch.manual_seed(0)
model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
state=torch.tensor(data.mean[None],dtype=torch.float32)
context=torch.zeros(1,304);exo=torch.zeros(1,1,144)
air=torch.tensor(data.mean_air[None,None],dtype=torch.float32)
pred=model(state,context,exo,air)
np.testing.assert_allclose(pred.detach().numpy()[0,0],data.mean,rtol=1e-5,atol=1e-5)
target=state[:,None]+.1
(pred-target).square().mean().backward()
gradient=model.forcing[-1].bias.grad[:5].abs().numpy()
assert np.all(gradient>1e-7)
broken=deepcopy(model);broken.zero_grad()
with torch.no_grad():broken.forcing[-1].bias[17:27]=-5
bad=broken(state,context,exo,air)
(bad-target).square().mean().backward()
assert float(bad[0,0,1])==0
assert float(broken.forcing[-1].bias.grad[1])==0
result={'fixed_initial_prediction':pred.detach().numpy()[0,0].tolist(),'expected_mean':data.mean.tolist(),
    'fixed_observed_output_gradient_magnitudes':gradient.tolist(),'old_bug_no3_prediction':float(bad[0,0,1]),
    'old_bug_no3_head_gradient':float(broken.forcing[-1].bias.grad[1]),'checks_passed':True,'test_evaluated':False}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
