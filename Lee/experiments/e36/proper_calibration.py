"""Optimize the actual probabilistic selection criterion on training-only dates."""
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
sys.path.insert(0,str(ROOT/'Lee/experiments/e32'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e26'))
from persistent_context import PersistentContextModel
from latent import Data,precompute,simulate,noise_paths,summarize,gate
from innovations import crps
from correlated import event_diagnostics


def proper(traj,true,sd):
    samples=traj.reshape(-1,*traj.shape[2:]);target=true[:,1:]
    concentration=float((crps(samples,target).mean((0,1))/sd).mean())
    pn=(samples[...,0]>4).mean(0);pd=(samples[...,2:].max(-1)>2).mean(0)
    return concentration+float(((pn-(target[...,0]>4))**2).mean())+float(((pd-(target[...,2:].max(-1)>2))**2).mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads((a.parent/'summary.json').read_text());assert source['model_kind']=='context'
    ref=json.loads(a.reference_summary.read_text());raw=pd.read_pickle(a.data);data=Data(raw,ref['fit_days'])
    models=[]
    for member in range(5):
        path=a.parent/f"epoch{source['selected_epoch']}_m{member}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['checkpoint_hashes'][path.name]
        model=PersistentContextModel(data.sd,data.mean,data.mean_air,data.hi)
        model.load_state_dict(torch.load(path,weights_only=True));model.eval();models.append(model)
    old_bias=np.array(source['bias']);scale=np.array(source['scale']);bank=np.load(a.parent/'residual_bank.npz')['residual']
    cal=ref['calibration_days'];cache=precompute(models,data,cal,16);noise=noise_paths(bank,cal)
    memo={};trace=[]
    def objective(bias):
        key=tuple(np.round(bias,12))
        if key not in memo:
            score=proper(simulate(cache,bias,noise,scale),cache['true'],data.sd)
            memo[key]=score;trace.append({'bias':bias.tolist(),'score':score})
        return memo[key]
    parent_score=objective(old_bias)
    np.testing.assert_allclose(parent_score,source['inner_scales'][source['selected_scale']]['proper_selection_score'],rtol=1e-12)
    zero=old_bias.copy();zero[[0,2,3,4]]=0
    bias=min([old_bias.copy(),zero],key=objective);bounds=np.array([.5,0,.25,.25,.25]);logs=[]
    for sweep in range(4):
        before=objective(bias)
        for output in [0,2,3,4]:
            if sweep<2:grid=np.linspace(-bounds[output],bounds[output],21)
            else:
                radius=2*bounds[output]/20/(4**(sweep-2))
                grid=np.clip(bias[output]+np.linspace(-radius,radius,9),-bounds[output],bounds[output])
            candidates=[]
            for value in np.unique(np.r_[grid,bias[output]]):
                candidate=bias.copy();candidate[output]=value;candidates.append(candidate)
            bias=min(candidates,key=objective)
        logs.append({'sweep':sweep,'score':objective(bias),'bias':bias.tolist(),'before':before})
        print('SWEEP',sweep,objective(bias),bias.tolist(),flush=True)
    best=min(trace,key=lambda row:row['score']);bias=np.array(best['bias'])
    inner=summarize(simulate(cache,bias,noise,scale),cache['true'],data.sd)
    np.testing.assert_allclose(inner['proper_selection_score'],best['score'],rtol=1e-12)
    days=raw['days']['val'];vc=precompute(models,data,days,16);vn=noise_paths(bank,days)
    control=summarize(simulate(vc,old_bias,vn,scale),vc['true'],data.sd)
    np.testing.assert_allclose(control['normalized_mae'],source['validation']['normalized_mae'],rtol=1e-12)
    traj=simulate(vc,bias,vn,scale);score=summarize(traj,vc['true'],data.sd)
    score['event_diagnostics']=event_diagnostics(traj,vc['true'])
    low=simulate(vc,bias,vn,scale,.75);high=simulate(vc,bias,vn,scale,1.25)
    score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
        'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
    sensitivity=summarize(simulate(vc,bias,noise_paths(bank,days,16),scale),vc['true'],data.sd)
    result={'selected_epoch':source['selected_epoch'],'selected_scale':source['selected_scale'],'bias':bias.tolist(),
        'scale':scale.tolist(),'model_kind':'context','state_row_sum_cap':.995,'gate':gate(score,ref['validation']['old']),
        'validation':score,'control_validation':control,'inner_selected':inner,'inner_parent_score':parent_score,
        'sixteen_draw_sensitivity':sensitivity,'search':{'unique_evaluations':len(trace),'sweeps':logs},
        'checkpoint_hashes':source['checkpoint_hashes'],'parent_runs':str(a.parent),'code_commit':commit,
        'parent_summary_sha256':hashlib.sha256((a.parent/'summary.json').read_bytes()).hexdigest(),'test_evaluated':False}
    (a.out/'calibration_trace.json').write_text(json.dumps(trace,indent=2,allow_nan=False))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('RESULT',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
