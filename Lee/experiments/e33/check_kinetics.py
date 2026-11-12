"""Check the kinetic scaling operator and its actual recursive implementation."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from kinetic_scaling import KineticScalingModel,Data,cache
from full_day import sequences
from latent import simulate

p=argparse.ArgumentParser();p.add_argument('--data',required=True)
p.add_argument('--reference-summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
with open(a.reference_summary) as f:ref=json.load(f)
data=Data(pd.read_pickle(a.data),ref['fit_days']);days=ref['fit_days'][:2]
model=KineticScalingModel(data.sd,data.mean,data.mean_air,data.hi,data.em[1],data.es[1])
V=torch.eye(7)[None]*.95;V[0,0,2:5]=-.01
f=torch.ones(1,7)*.1;f[0,0]=-.1
g=torch.ones(1,7,10)*.01;g[:,0]=-.01
for original,scaled in zip((V,f,g),model.scale_coefficients(V,f,g,torch.tensor([15.]))):
    np.testing.assert_allclose(original.numpy(),scaled.numpy(),rtol=1e-6,atol=1e-6)
values=[]
for temperature in [10.,20.]:
    v,force,gain=model.scale_coefficients(V,f,g,torch.tensor([temperature]))
    state=torch.ones(1,7);air=torch.ones(1,10)
    output=torch.einsum('bij,bj->bi',v,state)+force+(gain*air[:,None]).sum(-1)
    more=torch.einsum('bij,bj->bi',v,state)+force+(gain*(air*1.25)[:,None]).sum(-1)
    assert more[0,0]<=output[0,0] and torch.all(more[0,2:5]>=output[0,2:5])
    assert torch.all((v[:,0,0]>0)&(v[:,0,0]<1));values.append(float(output[0,0]))
assert values[0]>values[1]
s,c,e,air,target=sequences(data,days);prediction=model(s,c,e,air)
cached=simulate(cache([model],data,days,16),np.zeros(5))
np.testing.assert_allclose(cached[0,0],prediction.detach().numpy(),rtol=2e-4,atol=2e-4)
((prediction-target)/model.sd[:5]).square().mean().backward()
gradient=model.forcing[-1].bias.grad[:5].detach().numpy()
assert np.isfinite(gradient).all() and np.all(np.abs(gradient)>1e-7)
result={'reference15_identity':True,'cold_fixed_coefficient_retention':values,
    'air_direction_and_self_retention_checks':True,'full_day_torch_cache_agreement':True,
    'observed_head_gradients':gradient.tolist(),'test_evaluated':False}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
