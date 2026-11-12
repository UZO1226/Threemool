"""Check the root equation, physical directions, gradients and full cache path."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from monod_hybrid import MonodModel,positive_root,Data,cache
from full_day import sequences
from latent import simulate

p=argparse.ArgumentParser();p.add_argument('--data',required=True)
p.add_argument('--reference-summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
B=torch.tensor([0.,.1,1.,10.]);v=torch.ones(4)*.5;k=torch.tensor(1.)
x=positive_root(B,v,k)
np.testing.assert_allclose((x+v*x/(k+x)).numpy(),B.numpy(),rtol=1e-6,atol=1e-6)
assert torch.all(positive_root(B,v*2,k)<=x)
assert (B[2]-x[2])/B[2]>(B[3]-x[3])/B[3]
with open(a.reference_summary) as f:ref=json.load(f)
data=Data(pd.read_pickle(a.data),ref['fit_days']);days=ref['fit_days'][:2]
model=MonodModel(data.sd,data.mean,data.mean_air,data.hi,data.em[1],data.es[1])
mean=torch.tensor(data.mean[None],dtype=torch.float32)
equilibrium=model(mean,torch.zeros(1,304),torch.zeros(1,1,144),torch.tensor(data.mean_air[None,None],dtype=torch.float32))
np.testing.assert_allclose(equilibrium.detach().numpy()[0,0],data.mean,rtol=1e-5,atol=1e-5)
s,c,e,air,target=sequences(data,days);prediction=model(s,c,e,air)
np.testing.assert_allclose(simulate(cache([model],data,days,16),np.zeros(5))[0,0],prediction.detach().numpy(),rtol=2e-4,atol=2e-4)
more=model(s,c,e,torch.log1p(1.25*torch.expm1(air)))
assert float((more[...,0]-prediction[...,0]).max().detach())<1e-5
assert float((prediction[...,2:]-more[...,2:]).max().detach())<1e-5
((prediction-target)/model.sd[:5]).square().mean().backward()
gradient=model.kinetics[-1].bias.grad.detach().numpy()
assert np.isfinite(gradient).all() and np.all(np.abs(gradient)>1e-7)
result={'root_equation':True,'slower_fractional_clearance_at_high_nh4':True,'mean_equilibrium':True,
    'torch_numpy_full_day_agreement':True,'air_order':True,'reaction_head_gradients':gradient.tolist(),
    'test_evaluated':False}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
