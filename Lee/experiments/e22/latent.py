"""Contractive monotone latent dynamics trained on multi-step sequences."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e16'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e18'))
from surrogate_dw import STATE,EXO,ZONES
from cooperative import gate
from innovations import noise_paths,summarize


class LatentModel(nn.Module):
    def __init__(self,sd,mean,mean_air,hi):
        super().__init__()
        self.register_buffer('sd',torch.tensor(np.r_[sd,sd[0],sd[0]],dtype=torch.float32))
        self.register_buffer('sign',torch.tensor([-1,1,1,1,1,-1,-1],dtype=torch.float32))
        self.register_buffer('hi',torch.tensor(np.r_[hi,hi[0],hi[0]],dtype=torch.float32))
        self.forcing=nn.Sequential(nn.Linear(144,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,77))
        self.initializer=nn.Sequential(nn.Linear(304,64),nn.Tanh(),nn.Linear(64,2))
        mask=torch.ones(7,7,dtype=torch.bool);mask[:,1]=False;mask[1]=False;mask[1,1]=True
        self.register_buffer('mask',mask)
        w=torch.full((7,7),-8.);w.diagonal().fill_(8)
        w[0]=torch.full((7,),-8.);w[0,0]=float(np.log(.95));w[0,6]=float(np.log(.05))
        w[6]=torch.full((7,),-8.);w[6,6]=float(np.log(.9));w[6,5]=float(np.log(.1))
        self.matrix_logits=nn.Parameter(w)
        alpha=np.array([.95,.99,.2,.2,.2,.95,.95])/.995
        self.gate_base=nn.Parameter(torch.tensor(np.log(alpha/(1-alpha)),dtype=torch.float32))
        self.temp_raw=nn.Parameter(torch.tensor([-3.,0.,0.,0.,0.,-3.,-3.]))
        with torch.no_grad():
            self.forcing[-1].weight.zero_();self.forcing[-1].bias.zero_()
            self.forcing[-1].bias[7:]=-5.
            self.forcing[-1].bias[17:27]=0.
            scaled=torch.tensor(np.r_[mean,mean[0],mean[0]],dtype=torch.float32)*self.sign/self.sd
            A=.995*torch.sigmoid(self.gate_base)[:,None]*torch.softmax(w.masked_fill(~mask,-1e9),dim=-1)
            gain=torch.full((7,10),float(F.softplus(torch.tensor(-5.))))
            gain[1]=0
            self.forcing[-1].bias[:7]=scaled-A@scaled-gain@torch.tensor(mean_air,dtype=torch.float32)
            self.initializer[-1].weight.zero_()
            self.initializer[-1].bias.fill_(float(np.log(np.expm1(max(mean[0]/sd[0],.01)))))

    def initial(self,observed,context):
        latent=F.softplus(self.initializer(context))*self.sd[0]
        return torch.cat([observed,latent],-1)

    def coefficients(self,exo):
        out=self.forcing(exo)
        temp=exo[:,-8]  # final exogenous vector: q, temperature, sin, cos, ...
        effects=self.temp_raw.clone()
        effects[[0,5,6]]=-F.softplus(effects[[0,5,6]])
        gates=.995*torch.sigmoid(self.gate_base+temp[:,None]*effects)
        A=gates[:,:,None]*torch.softmax(self.matrix_logits.masked_fill(~self.mask,-1e9),-1)
        gain=F.softplus(out[:,7:].reshape(-1,7,10))
        gain=gain.clone();gain[:,1]=out[:,7:].reshape(-1,7,10)[:,1]
        conversion=self.sd*self.sign
        V=A*conversion[None,:,None]/conversion[None,None,:]
        force=out[:,:7]*conversion
        gains=gain*conversion[None,:,None]
        return V,force,gains

    def forward(self,state,context,exo,air):
        b,t,_=exo.shape
        V,f,g=self.coefficients(exo.reshape(-1,144))
        V=V.reshape(b,t,7,7);f=f.reshape(b,t,7);g=g.reshape(b,t,7,10)
        forcing=f+(g*air[:,:,None,:]).sum(-1)
        s=self.initial(state,context);pred=[]
        for k in range(t):
            s=torch.clamp(torch.einsum('bij,bj->bi',V[:,k],s)+forcing[:,k],min=0)
            s=torch.minimum(s,self.hi)
            pred.append(s[:,:5])
        return torch.stack(pred,1)


class Data:
    def __init__(self,data,fit_days):
        self.days=data['days'];self.table=data['data'].loc[:self.days['val'][-1]]
        self.state=self.table[STATE].to_numpy(float);self.exo=self.table[EXO].to_numpy(float)
        self.zone=self.table[ZONES].to_numpy(float)
        self.positions={day:self.table.index.get_indexer(data['data'].loc[day].index) for day in fit_days+self.days['train'][-30:]+self.days['val']}
        rows=np.concatenate([self.positions[day] for day in fit_days])
        self.sd=self.state[rows].std(0);self.mean=self.state[rows].mean(0);self.hi=self.state[rows].max(0)*1.1
        self.em=self.exo[rows].mean(0);self.es=self.exo[rows].std(0)+1e-6
        raw=np.c_[self.state,self.exo,self.zone/1000]
        self.cm=raw[rows].mean(0);self.cs=raw[rows].std(0)+1e-6;self.context=(raw-self.cm)/self.cs
        self.normal_exo=(self.exo-self.em)/self.es
        self.mean_air=np.log1p(np.c_[self.zone[rows],self.zone[np.maximum(rows-1,0)]]/1000).mean(0)

    def exohistory(self,positions,history):
        offsets=np.arange(-15,1) if history==16 else np.zeros(16,dtype=int)
        return self.normal_exo[np.maximum(positions[...,None]+offsets,0)].reshape(*positions.shape,144)

    def past(self,positions,history):
        offsets=np.arange(-16,0) if history==16 else np.full(16,-1)
        indices=np.maximum(positions[...,None]+offsets,0)
        assert np.all(indices<positions[...,None])
        return self.context[indices].reshape(*positions.shape,304)

    def chunks(self,days,history):
        starts=np.array([self.positions[day][k] for day in days for k in range(0,80,4)])
        day_starts=np.array([self.positions[day][0] for day in days for k in range(0,80,4)])
        p=starts[:,None]+np.arange(16)
        prev=np.maximum(p-1,day_starts[:,None])
        air=np.log1p(np.c_[self.zone[p].reshape(-1,5),self.zone[prev].reshape(-1,5)]/1000).reshape(len(starts),16,10)
        values=[self.state[starts],self.past(starts,history),self.exohistory(p,history),air,self.state[p+1]]
        return [torch.tensor(v,dtype=torch.float32) for v in values]


def train(data,fit_days,history,out):
    models=[];logs=[]
    for member in range(5):
        torch.manual_seed(member)
        model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        selected=np.random.RandomState(member).choice(len(fit_days),int(.8*len(fit_days)),replace=False)
        days=[fit_days[k] for k in selected]
        s,c,e,a,target=data.chunks(days,history)
        trace=[]
        for epoch in range(20):
            order=torch.randperm(len(s));total=0.
            for offset in range(0,len(order),64):
                idx=order[offset:offset+64]
                pred=model(s[idx],c[idx],e[idx],a[idx])
                error=(pred-target[idx])/model.sd[:5]
                loss=.5*(error.abs().mean()+error.square().mean())
                pn=torch.sigmoid((pred[:,:,0]-4)/.25);pd_=torch.sigmoid((pred[:,:,2:].max(-1).values-2)/.1)
                loss=loss+.5*((pn-(target[idx,:,0]>4).float()).square().mean()
                    +(pd_-(target[idx,:,2:].max(-1).values>2).float()).square().mean())
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                total+=float(loss.detach())*len(idx)
            trace.append(total/len(s))
            if (epoch+1)%5==0:print('TRAIN',history,member,epoch+1,trace[-1],flush=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval();torch.save(model.state_dict(),out/f'h{history}_m{member}.pt')
        models.append(model);logs.append(trace)
    return models,logs


def precompute(models,data,days,history,row_sum_bound=.995):
    positions=np.stack([data.positions[d] for d in days]);p=positions[:,:95]
    e=torch.tensor(data.exohistory(p,history).reshape(-1,144),dtype=torch.float32)
    c=torch.tensor(data.past(positions[:,0],history),dtype=torch.float32)
    s=torch.tensor(data.state[positions[:,0]],dtype=torch.float32)
    Vs=[];fs=[];gs=[];initial=[]
    with torch.no_grad():
        for model in models:
            V,f,g=model.coefficients(e)
            Vs.append(V.numpy().reshape(len(days),95,7,7).astype(float))
            fs.append(f.numpy().reshape(len(days),95,7).astype(float))
            gs.append(g.numpy().reshape(len(days),95,7,10).astype(float))
            initial.append(model.initial(s,c).numpy().astype(float))
    matrices=np.stack(Vs)
    conversion=np.r_[data.sd,data.sd[0],data.sd[0]]*np.array([-1,1,1,1,1,-1,-1])
    transformed=matrices*conversion[None,None,None,None,:]/conversion[None,None,None,:,None]
    assert np.isfinite(matrices).all() and transformed.min()>=-1e-8
    assert transformed.sum(-1).max()<=row_sum_bound+2e-7
    return {'V':matrices,'f':np.stack(fs),'g':np.stack(gs),'initial':np.stack(initial),
        'zone':data.zone[p],'previous_zone':np.concatenate([data.zone[p[:,:1]],data.zone[p[:,:-1]]],1),
        'true':data.state[positions],'hi':np.r_[data.hi,data.hi[0],data.hi[0]],'days':days}


def forcing(cache,air_scale):
    zone=cache['zone']*air_scale;prev=cache['previous_zone']*air_scale
    zone*=np.minimum(1.,9000/np.maximum(zone.sum(-1,keepdims=True),1))
    prev*=np.minimum(1.,9000/np.maximum(prev.sum(-1,keepdims=True),1))
    action=np.log1p(np.concatenate([zone,prev],-1)/1000)
    return cache['f']+np.einsum('mdtja,dta->mdtj',cache['g'],action)


def simulate(cache,bias,noise=None,scale=None,air_scale=1.):
    m,d,_=cache['initial'].shape;r=1 if noise is None else len(noise)//(m*d)
    state=np.repeat(cache['initial'][:,None],r,1)
    f=forcing(cache,air_scale);values=[]
    if noise is not None:noise=noise.reshape(m,r,d,95,5)
    padded=np.r_[bias,0,0]
    for t in range(95):
        state=np.einsum('mdjk,mrdk->mrdj',cache['V'][:,:,t],state)+f[:,None,:,t]+padded
        if noise is not None:state[...,:5]+=noise[:,:,:,t]*scale
        state=np.clip(state,0,cache['hi'])
        values.append(state[...,:5].copy())
    return np.stack(values,3)


def calibrate(cache,sd):
    target=cache['true'][:,1:];trace=[]
    tn=(target[...,0]>4).mean();td=(target[...,2:].max(-1)>2).mean()
    bound=np.array([.5,.5,.25,.25,.25])
    def objective(u):
        traj=simulate(cache,u*bound)
        mae=np.abs(traj-target[None,None]).mean((0,1,2,3))
        value=float((mae/sd).mean()+.5*(abs(expit((traj[...,0]-4)/.25).mean()-tn)
            +abs(expit((traj[...,2:].max(-1)-2)/.1).mean()-td)))
        trace.append({'bias':(u*bound).tolist(),'score':value});return value
    objective(np.zeros(5))
    opt=minimize(objective,np.zeros(5),method='Powell',bounds=[(-1,1)]*5,
        options={'maxfev':120,'maxiter':8,'xtol':.04,'ftol':.002})
    best=min(trace,key=lambda x:x['score'])
    return np.array(best['bias']),{'trace':trace,'success':bool(opt.success),'message':str(opt.message)}


def residual_bank(cache,bias):
    state=cache['initial'].copy();f=forcing(cache,1.);bank=[]
    for t in range(95):
        state[...,:5]=cache['true'][None,:,t]
        pred=np.einsum('mdjk,mdk->mdj',cache['V'][:,:,t],state)+f[:,:,t]+np.r_[bias,0,0]
        bank.append(cache['true'][None,:,t+1]-pred[...,:5]);state=np.clip(pred,0,cache['hi'])
    residual=np.stack(bank,2).reshape(len(state),-1,5)
    residual-=residual.mean(1,keepdims=True);residual[...,1]=0
    return residual


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    raw=pd.read_pickle(a.data);ref=json.loads(a.reference_summary.read_text())
    fit,cal=ref['fit_days'],ref['calibration_days'];data=Data(raw,fit)
    models={};caches={};biases={};metadata={};inner_mean={}
    for history in [1,16]:
        models[history],logs=train(data,fit,history,a.out)
        cache=precompute(models[history],data,cal,history);caches[history]=cache
        probe=precompute(models[history],data,cal[:1],history)
        pp=data.positions[cal[0]][:95][None]
        pe=torch.tensor(data.exohistory(pp,history),dtype=torch.float32)
        pc=torch.tensor(data.past(pp[:,0],history),dtype=torch.float32)
        ps=torch.tensor(data.state[pp[:,0]],dtype=torch.float32)
        pa=torch.tensor(np.log1p(np.concatenate([probe['zone'],probe['previous_zone']],-1)/1000),dtype=torch.float32)
        predicted=simulate(probe,np.zeros(5))
        with torch.no_grad():
            for m,model in enumerate(models[history]):
                np.testing.assert_allclose(predicted[m,0],model(ps,pc,pe,pa).numpy(),rtol=2e-4,atol=2e-4)
        bias,calibration=calibrate(cache,data.sd);biases[history]=bias
        value=summarize(simulate(cache,bias),cache['true'],data.sd)
        inner_mean[history]=value;metadata[history]={'fit_losses':logs,'bias':bias.tolist(),'calibration':calibration}
        (a.out/f'h{history}_metadata.json').write_text(json.dumps(metadata[history],indent=2))
        print('CALIBRATED',history,value['proper_selection_score'],flush=True)
    selected=min(inner_mean,key=lambda x:inner_mean[x]['proper_selection_score']);cache=caches[selected];bias=biases[selected]
    residual=residual_bank(cache,bias);np.savez_compressed(a.out/'residual_bank.npz',residual=residual)
    noise=noise_paths(residual,cal);inner={};scales={}
    for n in [0,.25,.5,1]:
        for d in [0,.5,1,1.5]:
            key=f'n{n}_d{d}';scale=np.array([n,0,d,d,d]);scales[key]=scale
            inner[key]=summarize(simulate(cache,bias,noise,scale),cache['true'],data.sd)
            print('INNER',key,inner[key]['proper_selection_score'],flush=True)
    winner=min(inner,key=lambda k:inner[k]['proper_selection_score']);scale=scales[winner]
    vc=precompute(models[selected],data,raw['days']['val'],selected);vn=noise_paths(residual,raw['days']['val'])
    score=summarize(simulate(vc,bias,vn,scale),vc['true'],data.sd)
    low=simulate(vc,bias,vn,scale,.75);high=simulate(vc,bias,vn,scale,1.25)
    score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
        'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
    sensitivity=summarize(simulate(vc,bias,noise_paths(residual,raw['days']['val'],draws=16),scale),vc['true'],data.sd)
    for value in metadata.values():value['calibration'].pop('trace',None)
    result={'selected_history':selected,'selected_scale':winner,'scale':scale.tolist(),'bias':bias.tolist(),
        'gate':gate(score,ref['validation']['old']),'validation':score,'sixteen_draw_sensitivity':sensitivity,
        'inner_mean':inner_mean,'inner_scales':inner,'metadata':metadata,'test_evaluated':False,
        'code_commit':commit,'verification':{'causal_past_context':True,'bounded_cooperative_matrix':True,'numpy_torch_rollout_agreement':True},
        'normalization':{'sd':data.sd.tolist(),'state_mean':data.mean.tolist(),'mean_air':data.mean_air.tolist(),'hi':data.hi.tolist(),
        'exo_mean':data.em.tolist(),'exo_sd':data.es.tolist(),
        'context_mean':data.cm.tolist(),'context_sd':data.cs.tolist()},
        'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.pt')}}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SELECTED',selected,winner,'GATE',json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
