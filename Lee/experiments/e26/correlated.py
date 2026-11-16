"""Select NH4 innovation correlation and scale without development fitting."""
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


def correlated_paths(residual,days,block,draws=8):
    out=noise_paths(residual,days,draws=draws)
    index=0;count=int(np.ceil(95/block));indices_checked=[]
    for m in range(len(residual)):
        for draw in range(draws):
            for day in days:
                seed=int.from_bytes(hashlib.sha256(f'20261112:{day}:{m}:{draw}'.encode()).digest()[:4],'little')
                rng=np.random.default_rng(seed)
                start=rng.integers(0,len(residual[m])//95,size=count)*95+rng.integers(0,96-block,size=count)
                indices=(start[:,None]+np.arange(block)).reshape(-1)[:95]
                assert np.all(start//95==(start+block-1)//95)
                out[index,:,0]=residual[m,indices,0];index+=1
    return out


def event_diagnostics(traj,true):
    result={};samples=traj.reshape(-1,*traj.shape[2:]);target=true[:,1:]
    for name,pred,actual in [('nh4',samples[...,0]>4,target[...,0]>4),
            ('do',samples[...,2:].max(-1)>2,target[...,2:].max(-1)>2)]:
        prob=pred.mean(0);positive=prob>=.5;tp=int((positive&actual).sum())
        fp=int((positive&~actual).sum());fn=int((~positive&actual).sum())
        reliability=[]
        for k in range(10):
            mask=(prob>=k/10)&((prob<(k+1)/10) if k<9 else (prob<=1))
            if mask.any():reliability.append({'lower':k/10,'count':int(mask.sum()),
                'mean_prediction':float(prob[mask].mean()),'observed_frequency':float(actual[mask].mean())})
        def runs(rows):
            count=0;lengths=[]
            for row in rows.reshape(-1,95):
                boundaries=np.diff(np.r_[False,row,False].astype(int));starts=np.flatnonzero(boundaries==1)
                ends=np.flatnonzero(boundaries==-1);lengths.extend((ends-starts).tolist());count+=len(starts)
            return {'runs_per_day':count/(rows.size/95),'mean_run_steps':float(np.mean(lengths)) if lengths else 0.}
        result[name]={'brier':float(((prob-actual)**2).mean()),'precision_at_half':tp/(tp+fp) if tp+fp else None,
            'recall_at_half':tp/(tp+fn) if tp+fn else None,'tp':tp,'fp':fp,'fn':fn,
            'reliability':reliability,'predicted_runs':runs(pred),'observed_runs':runs(actual)}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads((a.runs/'summary.json').read_text());ref=json.loads(a.reference_summary.read_text())
    raw=pd.read_pickle(a.data);data=Data(raw,ref['fit_days']);models=[]
    for member in range(5):
        path=a.runs/f"epoch{source['selected_epoch']}_m{member}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['checkpoint_hashes'][path.name]
        model=LatentModel(data.sd,data.mean,data.mean_air,data.hi)
        model.load_state_dict(torch.load(path,weights_only=True));model.eval();models.append(model)
    residual=np.load(a.runs/'residual_bank.npz')['residual'];bias=np.array(source['bias'])
    cal=ref['calibration_days'];cache=precompute(models,data,cal,16);inner={};settings={}
    np.testing.assert_array_equal(correlated_paths(residual,cal,4),noise_paths(residual,cal))
    for block in [1,4,16,48,95]:
        noise=correlated_paths(residual,cal,block)
        for n in [0,.5,1,1.5,2,3,4]:
            key=f'b{block}_n{n}';scale=np.array([n,0,1,1,1])
            score=summarize(simulate(cache,bias,noise,scale),cache['true'],data.sd)
            inner[key]=score;settings[key]=(block,n)
            print('INNER',key,score['proper_selection_score'],flush=True)
    selected=min(inner,key=lambda k:inner[k]['proper_selection_score'])
    block,n=settings[selected];print('SELECTED_ON_INNER',selected,flush=True)
    days=raw['days']['val'];cache=precompute(models,data,days,16);validation={}
    for key in dict.fromkeys(['b4_n1',selected]):
        b,s=settings[key];noise=correlated_paths(residual,days,b);scale=np.array([s,0,1,1,1])
        traj=simulate(cache,bias,noise,scale);score=summarize(traj,cache['true'],data.sd)
        score['event_diagnostics']=event_diagnostics(traj,cache['true'])
        low=simulate(cache,bias,noise,scale,.75);high=simulate(cache,bias,noise,scale,1.25)
        score['recursive_order']={'nh4_max_order_violation':float(np.maximum(high[...,0]-low[...,0],0).max()),
            'do_max_order_violation':float(np.maximum(low[...,2:]-high[...,2:],0).max())}
        validation[key]=score
    np.testing.assert_allclose(validation['b4_n1']['normalized_mae'],source['validation']['normalized_mae'],rtol=1e-12)
    scale=np.array([n,0,1,1,1]);traj=simulate(cache,bias,correlated_paths(residual,days,block,16),scale)
    result={'selected':selected,'gate':gate(validation[selected],ref['validation']['old']),
        'inner':inner,'validation':validation,'sixteen_draw_sensitivity':summarize(traj,cache['true'],data.sd),
        'selection_at_scale_upper_boundary':n==4,'test_evaluated':False,'code_commit':commit,
        'parent_summary_sha256':hashlib.sha256((a.runs/'summary.json').read_bytes()).hexdigest()}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('RESULT',selected,json.dumps(result['gate']),flush=True)


if __name__=='__main__':main()
