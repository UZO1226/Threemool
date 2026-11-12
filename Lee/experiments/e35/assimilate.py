"""Estimate per-episode bias using only already observed transition errors."""
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
sys.path.insert(0,str(ROOT/'Lee/experiments/e26'))
from latent import Data,LatentModel,precompute,simulate,noise_paths,summarize,gate
from correlated import event_diagnostics


def innovations(models,data,days,bias):
    starts=np.array([data.positions[day][0] for day in days])
    p=starts[:,None]+np.arange(-16,0)
    assert p.min()>0 and np.all(p+1<=starts[:,None])
    exo=torch.tensor(data.exohistory(p,16).reshape(-1,144),dtype=torch.float32)
    context=torch.tensor(data.past(p[:,0],16),dtype=torch.float32)
    physical=torch.tensor(data.state[p[:,0]],dtype=torch.float32)
    air=np.log1p(np.concatenate([data.zone[p],data.zone[p-1]],-1)/1000)
    result=[]
    with torch.no_grad():
        for model in models:
            V,f,g=model.coefficients(exo)
            V=V.numpy().reshape(len(days),16,7,7).astype(float)
            forcing=f.numpy().reshape(len(days),16,7).astype(float)
            gains=g.numpy().reshape(len(days),16,7,10).astype(float)
            forcing=forcing+(gains*air[:,:,None,:]).sum(-1)+np.r_[bias,0,0]
            state=model.initial(physical,context).numpy().astype(float);residual=[]
            for t in range(16):
                state[:,:5]=data.state[p[:,t]]
                prediction=np.einsum('djk,dk->dj',V[:,t],state)+forcing[:,t]
                residual.append(data.state[p[:,t]+1]-prediction[:,:5])
                state=np.clip(prediction,0,model.hi.numpy())
            result.append(np.stack(residual,1))
    residual=np.stack(result);mean=residual.mean(2);mean[...,1]=0
    bounded=np.clip(mean,-np.array([.5,0,.25,.25,.25]),np.array([.5,0,.25,.25,.25]))
    return bounded,{'earliest_history_position':int(p.min()),'last_target_not_after_start':True,
        'mean_residual_by_output':residual.mean((0,1,2)).tolist(),
        'rms_residual_by_output':np.sqrt((residual**2).mean((0,1,2))).tolist(),
        'clipped_fraction':float((np.abs(mean-bounded)>1e-12).mean())}


def corrected(cache,innovation,nh4_gain,do_gain):
    correction=innovation*np.array([nh4_gain,0,do_gain,do_gain,do_gain])
    out=dict(cache)
    out['f']=cache['f']+np.pad(correction,((0,0),(0,0),(0,2)))[:,:,None,:]
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads((a.parent/'summary.json').read_text());ref=json.loads(a.reference_summary.read_text())
    raw=pd.read_pickle(a.data);data=Data(raw,ref['fit_days']);models=[]
    for member in range(5):
        path=a.parent/f"epoch{source['selected_epoch']}_m{member}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['checkpoint_hashes'][path.name]
        model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
        model.load_state_dict(torch.load(path,weights_only=True));model.eval();models.append(model)
    bias=np.array(source['bias']);scale=np.array(source['scale'])
    bank=np.load(a.parent/'residual_bank.npz')['residual'];cal=ref['calibration_days']
    ci,cal_diag=innovations(models,data,cal,bias)
    # Negative control for target leakage: only the future is scrambled.
    probe=cal[:1];before,_=innovations(models,data,probe,bias)
    positions=data.positions[probe[0]][1:];saved=data.state[positions].copy()
    try:
        data.state[positions]=12345
        after,_=innovations(models,data,probe,bias)
        np.testing.assert_array_equal(before,after)
    finally:data.state[positions]=saved
    cache=precompute(models,data,cal,16);noise=noise_paths(bank,cal);inner={};settings={}
    for n in [0,.25,.5,1,2]:
        for d in [0,.25,.5,1]:
            key=f'n{n}_d{d}';settings[key]=(n,d)
            inner[key]=summarize(simulate(corrected(cache,ci,n,d),bias,noise,scale),cache['true'],data.sd)
            print('INNER',key,inner[key]['proper_selection_score'],flush=True)
    np.testing.assert_allclose(inner['n0_d0']['proper_selection_score'],source['inner_scales'][source['selected_scale']]['proper_selection_score'],rtol=1e-12)
    selected=min(inner,key=lambda key:inner[key]['proper_selection_score']);n,d=settings[selected]
    days=raw['days']['val'];vi,val_diag=innovations(models,data,days,bias)
    cache=precompute(models,data,days,16);noise=noise_paths(bank,days);validation={}
    trajectories={}
    for key in dict.fromkeys(['n0_d0',selected]):
        cn,cd=settings[key];adjusted=corrected(cache,vi,cn,cd)
        traj=simulate(adjusted,bias,noise,scale);trajectories[key]=traj
        score=summarize(traj,cache['true'],data.sd);score['event_diagnostics']=event_diagnostics(traj,cache['true'])
        low=simulate(adjusted,bias,noise,scale,.75);high=simulate(adjusted,bias,noise,scale,1.25)
        score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
            'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
        validation[key]=score
    np.testing.assert_allclose(validation['n0_d0']['normalized_mae'],source['validation']['normalized_mae'],rtol=1e-12)
    adjusted=corrected(cache,vi,n,d)
    sensitivity=summarize(simulate(adjusted,bias,noise_paths(bank,days,16),scale),cache['true'],data.sd)
    traj=trajectories[selected];truth=cache['true'][:,1:]
    daily={'nh4_bias_pp':(traj[...,0]>4).mean((0,1,3))*100-(truth[...,0]>4).mean(1)*100,
        'do_bias_pp':(traj[...,2:].max(-1)>2).mean((0,1,3))*100-(truth[...,2:].max(-1)>2).mean(1)*100}
    rng=np.random.default_rng(20261112);starts=rng.integers(0,len(days),size=(10000,8))
    idx=((starts[:,:,None]+np.arange(5))%len(days)).reshape(10000,-1)[:,:len(days)]
    result={'selected':selected,'gains':{'nh4':n,'do':d},'gate':gate(validation[selected],ref['validation']['old']),
        'inner':inner,'validation':validation,'sixteen_draw_sensitivity':sensitivity,
        'innovation_diagnostics':{'inner':cal_diag,'development':val_diag},
        'checks':{'future_target_scrambling_invariance':True,'zero_gain_inner_and_development_reconstruction':True},
        'date_block_intervals':{key:np.quantile(value[idx].mean(1),[.025,.975]).tolist() for key,value in daily.items()},
        'code_commit':commit,'test_evaluated':False,
        'parent_summary_sha256':hashlib.sha256((a.parent/'summary.json').read_bytes()).hexdigest()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',selected,json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
