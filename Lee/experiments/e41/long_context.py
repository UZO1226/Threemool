"""Test whether causal longer operating history identifies the daily regime."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e32'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e34'))
from persistent_context import PersistentContextModel
from joint_teacher import TeacherTransitions
from latent import Data
from full_day import main

WINDOWS=(96,288,672)
LONG_DIM=3*(3*19+1)


class LongContextData(Data):
    def __init__(self,*args):
        super().__init__(*args)
        self.prefix=np.vstack([np.zeros((1,19)),np.cumsum(self.context,axis=0)])
        self.square_prefix=np.vstack([np.zeros((1,19)),np.cumsum(self.context**2,axis=0)])

    def past(self,positions,history):
        short=super().past(positions,history)
        positions=np.asarray(positions,dtype=int)
        assert np.all(positions>0)
        blocks=[]
        for window in WINDOWS:
            start=np.maximum(positions-window,0);count=positions-start
            assert np.all(start<=positions-1)
            mean=(self.prefix[positions]-self.prefix[start])/count[...,None]
            second=(self.square_prefix[positions]-self.square_prefix[start])/count[...,None]
            std=np.sqrt(np.maximum(second-mean**2,0))
            trend=self.context[positions-1]-self.context[start]
            blocks.extend([mean,std,trend,(count/window)[...,None]])
        long=np.concatenate(blocks,axis=-1)
        assert long.shape[-1]==LONG_DIM
        return np.concatenate([short,long],axis=-1)


class LongContextModel(PersistentContextModel):
    def __init__(self,*args):
        super().__init__(*args)
        rng=torch.random.get_rng_state()
        self.long_encoder=nn.Sequential(nn.Linear(LONG_DIM,64),nn.Tanh(),nn.Linear(64,7))
        with torch.no_grad():
            self.long_encoder[-1].weight.zero_();self.long_encoder[-1].bias.zero_()
        torch.random.set_rng_state(rng)

    def initial(self,observed,context):
        return super().initial(observed,context[...,:304])

    def episode_coefficients(self,exo,context,steps):
        assert context.shape[-1]==304+LONG_DIM
        V,f,g=super().episode_coefficients(exo,context[...,:304],steps)
        offset=self.long_encoder(context[...,304:])*(self.sd*self.sign)
        return V,f+offset.repeat_interleave(steps,dim=0),g


if __name__=='__main__':
    main(model_class=LongContextModel,data_class=LongContextData,auxiliary_factory=TeacherTransitions)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result.update({'model_kind':'long_context','state_row_sum_cap':.995,
        'training_loss':'.5 recursive plus .5 teacher-forced observed states, latent carry and fixed initial context',
        'context_definition':'Past16 raw normalized rows plus strictly past24h/72h/7d mean, std, endpoint change and coverage, held fixed for95 future steps.',
        'long_context_windows':list(WINDOWS),'long_context_dimensions':LONG_DIM})
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
