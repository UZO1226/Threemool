"""A learned NH4 source and implicit saturating reaction, with surrogate DO."""
import json
import math
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e32'))
from persistent_context import PersistentContextModel
from latent import Data,precompute
from full_day import main


def positive_root(available,extent,knh):
    a=knh+extent-available;root=torch.sqrt(a*a+4*available*knh)
    return torch.where(a>=0,2*available*knh/torch.clamp(root+a,min=1e-8),.5*(root-a))


class MonodModel(PersistentContextModel):
    def __init__(self,sd,mean,mean_air,hi,temp_mean=15.,temp_sd=1.):
        super().__init__(sd,mean,mean_air,hi)
        rng=torch.random.get_rng_state()
        self.kinetics=nn.Sequential(nn.Linear(208,64),nn.SiLU(),nn.Linear(64,2))
        torch.random.set_rng_state(rng)
        self.raw_retention=nn.Parameter(torch.tensor(math.log((.98/.999)/(1-.98/.999))))
        self.raw_knh=nn.Parameter(torch.tensor(math.log(math.expm1(.99))))
        self.raw_kdo=nn.Parameter(torch.tensor(math.log(math.expm1(.39))))
        self.register_buffer('temp_mean',torch.tensor(float(temp_mean)))
        self.register_buffer('temp_sd',torch.tensor(float(temp_sd)))
        self.register_buffer('kappa',torch.tensor(math.log(.5/.3)/5))
        source=.12;oxygen=float(mean[2:5].mean());nh4=float(mean[0])
        extent=(source-.02*nh4)*(1+nh4)/nh4
        capacity=extent/(oxygen/(.4+oxygen))/math.exp(math.log(.5/.3)/5*(temp_mean-15))
        assert capacity>0
        with torch.no_grad():
            self.kinetics[-1].weight.zero_()
            self.kinetics[-1].bias.copy_(torch.tensor([math.log(math.expm1(source)),math.log(math.expm1(capacity))]))

    def reaction_parameters(self,exo,context,steps):
        embedding=self.episode_encoder[1](self.episode_encoder[0](context))
        raw=self.kinetics(torch.cat([exo,embedding.repeat_interleave(steps,0)],-1))
        temperature=exo[:,-8]*self.temp_sd+self.temp_mean
        return {'load':F.softplus(raw[:,0]).reshape(len(context),steps),
            'vmax':(F.softplus(raw[:,1])*torch.exp(self.kappa*(temperature-15))).reshape(len(context),steps),
            'retention':.999*torch.sigmoid(self.raw_retention),
            'knh':.01+F.softplus(self.raw_knh),'kdo':.01+F.softplus(self.raw_kdo)}

    def forward(self,state,context,exo,air):
        b,t,_=exo.shape;flat=exo.reshape(-1,144)
        V,f,g=self.episode_coefficients(flat,context,t);r=self.reaction_parameters(flat,context,t)
        V=V.reshape(b,t,7,7);f=f.reshape(b,t,7);g=g.reshape(b,t,7,10)
        forcing=f+(g*air[:,:,None,:]).sum(-1)
        s=self.initial(state,context);values=[]
        for k in range(t):
            backbone=torch.einsum('bij,bj->bi',V[:,k],s)+forcing[:,k]
            oxygen=torch.minimum(torch.clamp(backbone[:,2:5],min=0),self.hi[2:5]).mean(-1)
            extent=r['vmax'][:,k]*oxygen/(r['kdo']+oxygen)
            available=torch.clamp(r['retention']*s[:,0]+r['load'][:,k],min=0)
            nh4=positive_root(available,extent,r['knh'])
            new=torch.cat([nh4[:,None],backbone[:,1:]],-1)
            s=torch.minimum(torch.clamp(new,min=0),self.hi);values.append(s[:,:5])
        return torch.stack(values,1)


def cache(models,data,days,history):return precompute(models,data,days,history,row_sum_bound=None)


if __name__=='__main__':
    reference=Path(sys.argv[sys.argv.index('--reference-summary')+1])
    path=Path(sys.argv[sys.argv.index('--data')+1]);ref=json.loads(reference.read_text())
    data=Data(pd.read_pickle(path),ref['fit_days'])
    def factory(sd,mean,mean_air,hi):return MonodModel(sd,mean,mean_air,hi,data.em[1],data.es[1])
    main(model_class=factory,precompute_fn=cache)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result.update({'model_kind':'monod','state_row_sum_cap':None,
        'matrix_diagnostic_scope':'Linear backbone only, NH4 overridden by nonlinear implicit reaction',
        'reaction':'Positive learned source, learned positive Monod constants and temperature-scaled capacity'})
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
