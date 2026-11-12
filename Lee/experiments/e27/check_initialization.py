"""Compare actual initial coefficients and output gradients of the cap ablation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from slow_memory import SlowMemoryModel,LatentModel
from latent import Data

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
raw=pd.read_pickle(a.data);data=Data(raw,raw['days']['train'][:-30])
torch.manual_seed(0);old=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
torch.manual_seed(0);new=SlowMemoryModel(data.sd,data.mean,data.mean_air,data.hi)
exo=torch.zeros(1,144)
for x,y in zip(old.coefficients(exo),new.coefficients(exo)):
    np.testing.assert_allclose(x.detach().numpy(),y.detach().numpy(),rtol=1e-5,atol=1e-5)
s=torch.tensor(data.mean[None],dtype=torch.float32);c=torch.zeros(1,304)
air=torch.tensor(data.mean_air[None,None],dtype=torch.float32)
pred=new(s,c,exo[:,None],air)
np.testing.assert_allclose(pred.detach().numpy()[0,0],data.mean,rtol=1e-5,atol=1e-5)
(pred-s[:,None]-.1).square().mean().backward()
gradient=new.forcing[-1].bias.grad[:5].abs().numpy();assert np.all(gradient>1e-7)
result={'zero_temperature_coefficients_match':True,'equilibrium_matches':True,
    'all_observed_head_gradients_nonzero':True,'gradient_magnitudes':gradient.tolist(),
    'cap':new.cap,'test_evaluated':False}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
