"""Select an initial-NH4 anchor on inner proper score, retaining fixed e24."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
from latent import Data,LatentModel,precompute,simulate,noise_paths,summarize,gate


def anchor(traj,cache,weight):
    result=traj.copy()
    result[...,0]=(1-weight)*traj[...,0]+weight*cache['true'][None,None,:,0,0,None]
    return result


p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
source=json.loads((a.runs/'summary.json').read_text());ref=json.loads(a.reference_summary.read_text())
raw=pd.read_pickle(a.data);data=Data(raw,ref['fit_days']);models=[]
for member in range(5):
    model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
    model.load_state_dict(torch.load(a.runs/f"epoch{source['selected_epoch']}_m{member}.pt",weights_only=True))
    model.eval();models.append(model)
residual=np.load(a.runs/'residual_bank.npz')['residual'];bias=np.array(source['bias']);scale=np.array(source['scale'])
cal=ref['calibration_days'];cache=precompute(models,data,cal,16)
traj=simulate(cache,bias,noise_paths(residual,cal),scale)
inner={str(k/10):summarize(anchor(traj,cache,k/10),cache['true'],data.sd) for k in range(11)}
selected=min(inner,key=lambda k:inner[k]['proper_selection_score']);weight=float(selected)
print('SELECTED_ON_INNER',selected,flush=True)
days=raw['days']['val'];cache=precompute(models,data,days,16);noise=noise_paths(residual,days)
traj=simulate(cache,bias,noise,scale);scores={}
for key in dict.fromkeys(['0.0',selected,'1.0']):
    scores[key]=summarize(anchor(traj,cache,float(key)),cache['true'],data.sd)
np.testing.assert_allclose(scores['0.0']['normalized_mae'],source['validation']['normalized_mae'],rtol=1e-12)
low=anchor(simulate(cache,bias,noise,scale,.75),cache,weight)
high=anchor(simulate(cache,bias,noise,scale,1.25),cache,weight)
scores[selected]['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
    'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
sensitivity=summarize(anchor(simulate(cache,bias,noise_paths(residual,days,draws=16),scale),cache,weight),cache['true'],data.sd)
result={'selected_weight':weight,'inner':inner,'validation':scores,'gate':gate(scores[selected],ref['validation']['old']),
    'sixteen_draw_sensitivity':sensitivity,'test_evaluated':False,'code_commit':commit,
    'parent_summary_sha256':hashlib.sha256((a.runs/'summary.json').read_bytes()).hexdigest()}
(a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({'weight':weight,'gate':result['gate'],'nh4':scores[selected]['nh4_violation'],'do':scores[selected]['do_violation']}),flush=True)
