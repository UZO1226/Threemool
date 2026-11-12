"""A stated kinetic prior, not a calibrated ASM1 process model."""
import json
import math
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e23'))
from latent import Data,LatentModel,precompute
from full_day import main


class KineticScalingModel(LatentModel):
    def __init__(self,sd,mean,mean_air,hi,temp_mean=15.,temp_sd=1.):
        super().__init__(sd,mean,mean_air,hi)
        self.register_buffer('temp_mean',torch.tensor(float(temp_mean)))
        self.register_buffer('temp_sd',torch.tensor(float(temp_sd)))
        self.register_buffer('kappa',torch.tensor(math.log(.5/.3)/5))

    def scale_coefficients(self,V,f,g,temperature):
        rate=torch.exp(self.kappa*(temperature-15))
        scaledV=V.clone();scaledf=f.clone();scaledg=g.clone()
        scaledV[:,0]=torch.where(V[:,0]<0,V[:,0]*rate[:,None],V[:,0])
        scaledV[:,0,0]=V[:,0,0].clamp(1e-8,1).pow(rate)
        scaledf[:,0]=torch.where(f[:,0]<0,f[:,0]*rate,f[:,0])
        scaledg[:,0]=g[:,0]*rate[:,None]
        return scaledV,scaledf,scaledg

    def coefficients(self,exo):
        V,f,g=super().coefficients(exo)
        temperature=exo[:,-8]*self.temp_sd+self.temp_mean
        return self.scale_coefficients(V,f,g,temperature)


def cache(models,data,days,history):
    return precompute(models,data,days,history,row_sum_bound=None)


if __name__=='__main__':
    data_path=Path(sys.argv[sys.argv.index('--data')+1])
    reference=Path(sys.argv[sys.argv.index('--reference-summary')+1])
    ref=json.loads(reference.read_text());data=Data(pd.read_pickle(data_path),ref['fit_days'])
    def factory(sd,mean,mean_air,hi):
        return KineticScalingModel(sd,mean,mean_air,hi,data.em[1],data.es[1])
    main(model_class=factory,precompute_fn=cache)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result.update({'model_kind':'kinetic','state_row_sum_cap':None,
        'base_state_row_sum_cap':.995,'fixed_kappa':math.log(.5/.3)/5,
        'interpretation':'Empirical kinetic temperature scaling, not a full ASM1 or calibrated Tilburg model.'})
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
