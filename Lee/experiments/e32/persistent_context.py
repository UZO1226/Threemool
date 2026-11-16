"""Keep a strictly pre-episode context offset throughout recursive simulation."""
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e23'))
from latent import LatentModel
from full_day import main


class PersistentContextModel(LatentModel):
    def __init__(self,*args):
        super().__init__(*args)
        rng=torch.random.get_rng_state()
        self.episode_encoder=nn.Sequential(nn.Linear(304,64),nn.Tanh(),nn.Linear(64,7))
        with torch.no_grad():
            self.episode_encoder[-1].weight.zero_();self.episode_encoder[-1].bias.zero_()
        torch.random.set_rng_state(rng)

    def episode_coefficients(self,exo,context,steps):
        V,f,g=self.coefficients(exo)
        assert len(exo)==len(context)*steps
        offset=self.episode_encoder(context)*(self.sd*self.sign)
        return V,f+offset.repeat_interleave(steps,dim=0),g

    def forward(self,state,context,exo,air):
        b,t,_=exo.shape
        V,f,g=self.episode_coefficients(exo.reshape(-1,144),context,t)
        V=V.reshape(b,t,7,7);f=f.reshape(b,t,7);g=g.reshape(b,t,7,10)
        forcing=f+(g*air[:,:,None,:]).sum(-1)
        s=self.initial(state,context);pred=[]
        for k in range(t):
            s=torch.clamp(torch.einsum('bij,bj->bi',V[:,k],s)+forcing[:,k],min=0)
            s=torch.minimum(s,self.hi);pred.append(s[:,:5])
        return torch.stack(pred,1)


if __name__=='__main__':
    main(model_class=PersistentContextModel)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result['model_kind']='context';result['state_row_sum_cap']=.995
    result['context_definition']='Strictly past16 observed intervals, held fixed for95 future steps.'
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
