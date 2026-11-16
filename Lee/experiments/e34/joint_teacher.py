"""Preserve episode conditioning while jointly supervising local transitions."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e32'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e31'))
from persistent_context import PersistentContextModel
from joint_horizon import point_loss
from full_day import main,sequences


class TeacherTransitions:
    def __init__(self,data,days):
        positions=np.stack([data.positions[d][:95] for d in days])
        self.state=torch.tensor(data.state[positions],dtype=torch.float32)
        _,self.context,self.exo,self.air,self.target=sequences(data,days)
        np.testing.assert_array_equal(self.context.numpy(),data.past(positions[:,0],16).astype(np.float32))

    def predictions(self,model,idx):
        observed=self.state[idx];context=self.context[idx];exo=self.exo[idx];air=self.air[idx]
        b,t,_=exo.shape
        V,f,g=model.episode_coefficients(exo.reshape(-1,144),context,t)
        V=V.reshape(b,t,7,7);f=f.reshape(b,t,7);g=g.reshape(b,t,7,10)
        forcing=f+(g*air[:,:,None,:]).sum(-1)
        state=model.initial(observed[:,0],context);values=[]
        for k in range(t):
            teacher=torch.cat([observed[:,k],state[:,5:]],-1)
            state=torch.einsum('bij,bj->bi',V[:,k],teacher)+forcing[:,k]
            state=torch.minimum(torch.clamp(state,min=0),model.hi)
            values.append(state[:,:5])
        return torch.stack(values,1)

    def __call__(self,model,idx):
        return point_loss(self.predictions(model,idx),self.target[idx],model)


if __name__=='__main__':
    main(model_class=PersistentContextModel,auxiliary_factory=TeacherTransitions)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result.update({'model_kind':'context','state_row_sum_cap':.995,
        'training_loss':'.5 recursive plus .5 teacher-forced observed states, latent carry and fixed initial context',
        'context_definition':'Strictly past16 observed intervals, held fixed for95 future steps.'})
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
