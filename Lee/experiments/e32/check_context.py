"""Check zero-offset equivalence, RNG pairing and learnable context offsets."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from persistent_context import PersistentContextModel,LatentModel
from full_day import sequences
from latent import Data,precompute,simulate

p=argparse.ArgumentParser();p.add_argument('--data',required=True)
p.add_argument('--reference-summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1)
with open(a.reference_summary) as f:ref=json.load(f)
data=Data(pd.read_pickle(a.data),ref['fit_days']);days=ref['fit_days'][:2]
torch.manual_seed(0);base=LatentModel(data.sd,data.mean,data.mean_air,data.hi);base_rng=torch.random.get_rng_state()
torch.manual_seed(0);model=PersistentContextModel(data.sd,data.mean,data.mean_air,data.hi)
assert torch.equal(base_rng,torch.random.get_rng_state())
s,c,e,air,target=sequences(data,days)
prediction=model(s,c,e,air)
np.testing.assert_allclose(prediction.detach().numpy(),base(s,c,e,air).detach().numpy(),rtol=1e-5,atol=1e-5)
cache=precompute([model],data,days,16)
np.testing.assert_allclose(simulate(cache,np.zeros(5))[0,0],prediction.detach().numpy(),rtol=2e-4,atol=2e-4)
((prediction-target)/model.sd[:5]).square().mean().backward()
gradient=float(model.episode_encoder[-1].weight.grad.abs().sum())
assert np.isfinite(gradient) and gradient>1e-6
result={'full_day_zero_offset_equivalence':True,'rng_state_pairing':True,
    'cached_recursive_agreement':True,'context_head_gradient_sum':gradient,'test_evaluated':False}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
