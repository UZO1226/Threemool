"""Check rare-event gradient emphasis and the ordinary-regression limit."""
import argparse
import json

import numpy as np
import torch
from tail_weighted import weighted_regression

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(1);sd=torch.ones(7)
target=torch.zeros(1,2,5);target[0,:,0]=torch.tensor([1.,5.])
pred=target.clone();pred[0,:,0]+=1;pred.requires_grad_()
value=weighted_regression(pred,target,sd);value.backward()
gradient=pred.grad[0,:,0].numpy();ratio=float(gradient[1]/gradient[0])
np.testing.assert_allclose(ratio,4,rtol=1e-6);np.testing.assert_allclose(float(value),.2,rtol=1e-6)
target=torch.zeros(2,3,5);pred=torch.linspace(-1,1,30).reshape(2,3,5)
expected=.5*((pred-target).abs().mean()+(pred-target).square().mean())
np.testing.assert_allclose(float(weighted_regression(pred,target,sd)),float(expected),rtol=1e-6)
result={'rare_to_ordinary_gradient_ratio':ratio,'equal_error_regression':float(value),
    'no_event_baseline_equivalence':True,'checks_passed':True}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
