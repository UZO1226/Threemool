"""DO 및 NH4 분기의 교차 결합과 실측 상태 주입으로 오차를 분리한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from trial import load_model, routing_frame, arrays, pack, replay, score, write
from history_trial import features


@torch.no_grad()
def intervention(dm,nm,x,aux,do_routing,nh4_routing,oracle='none'):
    state,exog=pack(x);state=torch.tensor(state).float();exog=torch.tensor(exog).float()
    aux=torch.tensor(aux).float()
    pred=state[:,12].clone();result=[]
    for t in range(12,287):
        current=pred.clone()
        if oracle in ('nh4','both'):current[:,0]=state[:,t,0]
        if oracle in ('do','both'):current[:,1]=state[:,t,1]
        extra=aux[:,t].reshape(-1,4,2)
        xd=features(current,exog[:,t],extra if do_routing else torch.zeros_like(extra))
        dn=(current[:,1]+dm.do_net((xd-dm.do_mu)/dm.do_sd)[:,0]*dm.delta_sd[1]+dm.delta_mu[1]).clamp_min(0)
        if oracle in ('do','both'):dn=state[:,t+1,1]
        xn=features(current,exog[:,t],extra if nh4_routing else torch.zeros_like(extra),dn)
        nn=(current[:,0]+nm.nh4_net((xn-nm.nh4_mu)/nm.nh4_sd)[:,0]*nm.delta_sd[0]+nm.delta_mu[0]).clamp_min(0)
        pred=torch.stack((nn,dn),-1);assert torch.isfinite(pred).all()
        result.append(pred.numpy())
    return np.stack(result,1)


def main(a):
    torch.set_num_threads(1)
    if a.out.exists():raise ValueError(f'Refusing overwrite {a.out}')
    trial=json.loads((a.trial/'results.json').read_text());assert 'completed_at' in trial
    data=pd.read_pickle(a.data)
    parts={'inner_holdout':data['days']['train'][264:],'val':data['days']['val']}
    xs={s:arrays({'data':data['data'],'days':parts},s) for s in parts}
    frame=routing_frame(a.raw,end='2024-02-29')
    aux={s:np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in parts[s]]) for s in parts}
    r={'started_at':datetime.now(timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.raw,a.data,a.trial/'results.json')},
       'test_used':False,'e12_models_outer_val_used':False,'rows':[],'decomposition':[],'models':{}}
    for seed in (0,1):
        old=load_model(a.pretrained/f'without_routing_s{seed}.pt')
        new=load_model(a.pretrained/f'with_routing_s{seed}.pt')
        for split,x in xs.items():
            for d,dm in enumerate((old,new)):
                for n,nm in enumerate((old,new)):
                    pred=intervention(dm,nm,x,aux[split],d==1,n==1)
                    if d==n:
                        native=replay(dm,x,aux[split],'with_routing' if d else 'without_routing',275)
                        np.testing.assert_array_equal(native,pred)
                    r['rows'].append({'kind':'crossover','seed':seed,'split':split,'do_new':bool(d),'nh4_new':bool(n),**score(pred,x)})
            for kind,key in [('nh4','step_recall'),('nh4','bias'),('nh4','fp'),('do','mae')]:
                def f(d,n):return next(z[kind][key] for z in r['rows'] if z['kind']=='crossover' and z['seed']==seed and z['split']==split and z['do_new']==bool(d) and z['nh4_new']==bool(n))
                r['decomposition'].append({'seed':seed,'split':split,'metric':kind+'.'+key,
                    'baseline':f(0,0),'both_new':f(1,1),
                    'do_order_averaged_contribution':float(((f(1,0)-f(0,0))+(f(1,1)-f(0,1)))/2),
                    'nh4_order_averaged_contribution':float(((f(0,1)-f(0,0))+(f(1,1)-f(1,0)))/2),
                    'interaction':float(f(1,1)-f(1,0)-f(0,1)+f(0,0))})
        for label,path,routed in [('e11_without',a.pretrained/f'without_routing_s{seed}.pt',False),
                                  ('e11_with',a.pretrained/f'with_routing_s{seed}.pt',True),
                                  ('e12_single_mse',a.trial/f'single_mse_s{seed}.pt',True),
                                  ('e12_multi_boundary',a.trial/f'multi_boundary_s{seed}.pt',True)]:
            model=load_model(path);r['models'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for split,x in xs.items():
                if label.startswith('e12') and split=='val':continue
                for oracle in ('none','do','nh4','both'):
                    pred=intervention(model,model,x,aux[split],routed,routed,oracle)
                    r['rows'].append({'kind':'oracle','label':label,'seed':seed,'split':split,'oracle':oracle,**score(pred,x)})
    r['checks']={'native_crossover_identity':True,'finite_predictions':True}
    r['completed_at']=datetime.now(timezone.utc).isoformat();write(a.out,r)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','data','pretrained','trial','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
