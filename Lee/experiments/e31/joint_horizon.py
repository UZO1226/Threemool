"""Joint observed-state local fitting and fully recursive day fitting."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e23'))
from full_day import main,sequences


def point_loss(pred,target,model):
    error=(pred-target)/model.sd[:5]
    pn=torch.sigmoid((pred[:,:,0]-4)/.25)
    pd=torch.sigmoid((pred[:,:,2:].max(-1).values-2)/.1)
    return .5*(error.abs().mean()+error.square().mean())+.5*(
        (pn-(target[:,:,0]>4).float()).square().mean()
        +(pd-(target[:,:,2:].max(-1).values>2).float()).square().mean())


class LocalTransitions:
    def __init__(self,data,days):
        positions=np.stack([data.positions[day][:95] for day in days])
        self.state=torch.tensor(data.state[positions],dtype=torch.float32)
        self.context=torch.tensor(data.past(positions,16),dtype=torch.float32)
        _,_,self.exo,self.air,self.target=sequences(data,days)
        assert self.state.shape==self.target.shape and self.context.shape[-1]==304

    def predictions(self,model,idx):
        n=len(idx)
        return model(self.state[idx].reshape(-1,5),self.context[idx].reshape(-1,304),
            self.exo[idx].reshape(-1,1,144),self.air[idx].reshape(-1,1,10)).reshape(n,95,5)

    def __call__(self,model,idx):
        return point_loss(self.predictions(model,idx),self.target[idx],model)


if __name__=='__main__':
    main(auxiliary_factory=LocalTransitions)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result['training_loss']='.5 original full-day plus .5 observed-state one-step loss'
    result['state_row_sum_cap']=.995
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
