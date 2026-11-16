"""Emphasize observed safety tails during fitting, leaving evaluation fixed."""
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e23'))
from full_day import main


def weighted_regression(pred,target,sd):
    error=(pred-target)/sd[:5]
    weight=torch.ones_like(error)
    weight[:,:,0]+=3*(target[:,:,0]>4).float()
    weight[:,:,2:]+=3*(target[:,:,2:].max(-1).values>2).float()[:,:,None]
    weight=weight/weight.mean((0,1),keepdim=True)
    return .5*((weight*error.abs()).mean()+(weight*error.square()).mean())


def loss(pred,target,model):
    regression=weighted_regression(pred,target,model.sd)
    pn=torch.sigmoid((pred[:,:,0]-4)/.25)
    pd=torch.sigmoid((pred[:,:,2:].max(-1).values-2)/.1)
    return regression+.5*((pn-(target[:,:,0]>4).float()).square().mean()
        +(pd-(target[:,:,2:].max(-1).values>2).float()).square().mean())


if __name__=='__main__':
    main(loss_function=loss)
    out=Path(sys.argv[sys.argv.index('--out')+1]);path=out/'summary.json'
    result=json.loads(path.read_text());result['training_loss']='per-output normalized event regression weights4, original soft-Brier unchanged'
    result['state_row_sum_cap']=.995
    path.write_text(json.dumps(result,indent=2,allow_nan=False))
