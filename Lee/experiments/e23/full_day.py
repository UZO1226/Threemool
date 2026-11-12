"""Train complete-day latent dynamics and select shared checkpoints internally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/experiments/e22'))
from latent import Data,LatentModel,precompute,calibrate,simulate,residual_bank,noise_paths,summarize,gate


def sequences(data,days):
    positions=np.stack([data.positions[d] for d in days]);p=positions[:,:95]
    zone=data.zone[p];prev=np.concatenate([zone[:,:1],zone[:,:-1]],1)
    values=[data.state[positions[:,0]],data.past(positions[:,0],16),data.exohistory(p,16),
        np.log1p(np.concatenate([zone,prev],-1)/1000),data.state[positions[:,1:]]]
    return [torch.tensor(v,dtype=torch.float32) for v in values]


def main(model_class=LatentModel,precompute_fn=precompute,score_sd=None,
        epochs=80,checkpoints=(20,40,80),warm_start_dir=None,
        warm_start_epoch=80,learning_rate=.001,loss_function=None,
        auxiliary_factory=None):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    raw=pd.read_pickle(a.data);ref=json.loads(a.reference_summary.read_text())
    fit,cal=ref['fit_days'],ref['calibration_days'];data=Data(raw,fit);logs={}
    scoring_sd=data.sd if score_sd is None else np.asarray(score_sd)
    for member in range(5):
        torch.manual_seed(member);model=model_class(data.sd,data.mean,data.mean_air,data.hi)
        if warm_start_dir is not None:
            model.load_state_dict(torch.load(Path(warm_start_dir)/f'epoch{warm_start_epoch}_m{member}.pt',weights_only=True))
        opt=torch.optim.Adam(model.parameters(),lr=learning_rate)
        if 0 in checkpoints:torch.save(model.state_dict(),a.out/f'epoch0_m{member}.pt')
        ids=np.random.RandomState(member).choice(len(fit),int(.8*len(fit)),replace=False)
        days=[fit[k] for k in ids];s,c,e,air,target=sequences(data,days);trace=[]
        auxiliary=None if auxiliary_factory is None else auxiliary_factory(data,days)
        for epoch in range(1,epochs+1):
            order=torch.randperm(len(s));total=0.
            for k in range(0,len(order),16):
                idx=order[k:k+16];pred=model(s[idx],c[idx],e[idx],air[idx])
                error=(pred-target[idx])/model.sd[:5]
                loss=.5*(error.abs().mean()+error.square().mean())
                pn=torch.sigmoid((pred[:,:,0]-4)/.25);pd_=torch.sigmoid((pred[:,:,2:].max(-1).values-2)/.1)
                loss=loss+.5*((pn-(target[idx,:,0]>4).float()).square().mean()
                    +(pd_-(target[idx,:,2:].max(-1).values>2).float()).square().mean())
                if loss_function is not None:loss=loss_function(pred,target[idx],model)
                if auxiliary is not None:loss=.5*loss+.5*auxiliary(model,idx)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                total+=float(loss.detach())*len(idx)
            trace.append(total/len(s))
            assert all(torch.isfinite(value).all() for value in model.parameters())
            if epoch%10==0:print('TRAIN',member,epoch,trace[-1],flush=True)
            if epoch in checkpoints:torch.save(model.state_dict(),a.out/f'epoch{epoch}_m{member}.pt')
        logs[str(member)]=trace;(a.out/'train_losses.json').write_text(json.dumps(logs,indent=2))
    inner_mean={};metadata={};caches={};biases={};ensembles={}
    for epoch in checkpoints:
        models=[]
        for member in range(5):
            model=model_class(data.sd,data.mean,data.mean_air,data.hi)
            model.load_state_dict(torch.load(a.out/f'epoch{epoch}_m{member}.pt',weights_only=True));model.eval();models.append(model)
        cache=precompute_fn(models,data,cal,16);bias,calibration=calibrate(cache,scoring_sd)
        inner_mean[epoch]=summarize(simulate(cache,bias),cache['true'],scoring_sd)
        metadata[epoch]=calibration;caches[epoch]=cache;biases[epoch]=bias;ensembles[epoch]=models
        (a.out/f'epoch{epoch}_calibration.json').write_text(json.dumps(calibration,indent=2))
        print('CHECKPOINT',epoch,inner_mean[epoch]['proper_selection_score'],flush=True)
    epoch=min(inner_mean,key=lambda k:inner_mean[k]['proper_selection_score']);cache=caches[epoch];bias=biases[epoch]
    residual=residual_bank(cache,bias);np.savez_compressed(a.out/'residual_bank.npz',residual=residual)
    noise=noise_paths(residual,cal);inner={};scales={}
    for n in [0,.25,.5,1]:
        for d in [0,.5,1,1.5]:
            key=f'n{n}_d{d}';scale=np.array([n,0,d,d,d]);scales[key]=scale
            inner[key]=summarize(simulate(cache,bias,noise,scale),cache['true'],scoring_sd)
            print('INNER',key,inner[key]['proper_selection_score'],flush=True)
    selected=min(inner,key=lambda k:inner[k]['proper_selection_score']);scale=scales[selected]
    vc=precompute_fn(ensembles[epoch],data,raw['days']['val'],16);vn=noise_paths(residual,raw['days']['val'])
    score=summarize(simulate(vc,bias,vn,scale),vc['true'],scoring_sd)
    low=simulate(vc,bias,vn,scale,.75);high=simulate(vc,bias,vn,scale,1.25)
    score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
        'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
    sensitivity=summarize(simulate(vc,bias,noise_paths(residual,raw['days']['val'],draws=16),scale),vc['true'],scoring_sd)
    for value in metadata.values():value.pop('trace',None)
    result={'selected_epoch':epoch,'selected_scale':selected,'scale':scale.tolist(),'bias':bias.tolist(),
        'gate':gate(score,ref['validation']['old']),'validation':score,'sixteen_draw_sensitivity':sensitivity,
        'inner_mean':inner_mean,'inner_scales':inner,'calibration':metadata,'fit_losses':logs,
        'test_evaluated':False,'code_commit':commit,
        'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.pt')}}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',epoch,selected,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
