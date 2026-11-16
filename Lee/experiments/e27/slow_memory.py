"""Allow slightly amplifying state memory without changing response signs."""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e23'))
from latent import LatentModel,precompute
from full_day import main as train_full_day


class SlowMemoryModel(LatentModel):
    cap=1.05

    def __init__(self,*args):
        super().__init__(*args)
        with torch.no_grad():
            probability=.995*torch.sigmoid(self.gate_base)/self.cap
            self.gate_base.copy_(torch.logit(probability))

    def coefficients(self,exo):
        out=self.forcing(exo);temp=exo[:,-8];effects=self.temp_raw.clone()
        effects[[0,5,6]]=-F.softplus(effects[[0,5,6]])
        gates=self.cap*torch.sigmoid(self.gate_base+temp[:,None]*effects)
        A=gates[:,:,None]*torch.softmax(self.matrix_logits.masked_fill(~self.mask,-1e9),-1)
        gain=F.softplus(out[:,7:].reshape(-1,7,10)).clone()
        gain[:,1]=out[:,7:].reshape(-1,7,10)[:,1]
        conversion=self.sd*self.sign
        return (A*conversion[None,:,None]/conversion[None,None,:],out[:,:7]*conversion,gain*conversion[None,:,None])


def cache(models,data,days,history):
    result=precompute(models,data,days,history,row_sum_bound=SlowMemoryModel.cap)
    conversion=models[0].sd.numpy()*models[0].sign.numpy()
    transformed=result['V']*conversion[None,None,None,None,:]/conversion[None,None,None,:,None]
    print('MATRIX_DIAGNOSTIC',days[0],float(transformed.sum(-1).max()),flush=True)
    return result


if __name__=='__main__':
    train_full_day(model_class=SlowMemoryModel,precompute_fn=cache)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result['state_row_sum_cap']=SlowMemoryModel.cap
    result['only_changed_factor']='state matrix gate cap and equivalent zero-temperature initialization'
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
