"""고정 e35 전이 위에서 e54 불확실성 비용 신호의 탐지/FPR을 대조한다."""
import argparse
from datetime import datetime,timezone
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sklearn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EXP=HERE.parents[1]
sys.path.insert(0,str(EXP/'e45'))
from evaluate import load_selected
from darrow_env import BatchEnv,Dynamics,all_episodes,action_to_air
from common import sha,save

NAMES=['plain','fixed','original','candidate']


def counts(flag,truth):
    assert flag.shape==truth.shape
    return np.stack([(flag&truth).sum((1,2)),(flag&~truth).sum((1,2)),truth.sum((1,2)),(~truth).sum((1,2))],1).astype(float)


def rates(c):
    tp,fp,pos,neg=c.sum(0)
    return dict(recall_pct=float(100*tp/pos),fpr_pct=float(100*fp/neg),
                precision_pct=float(100*tp/max(1,tp+fp)),cost=float((tp+fp)/len(c)/5))


def block_intervals(candidate,reference):
    assert candidate.shape==reference.shape
    n=len(candidate);rng=np.random.default_rng(20261112)
    starts=rng.integers(n,size=(2000,(n+6)//7))
    index=((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
    a=candidate[index].sum(1);b=reference[index].sum(1)
    d=np.column_stack([100*(a[:,0]/a[:,2]-b[:,0]/b[:,2]),100*(a[:,1]/a[:,3]-b[:,1]/b[:,3])])
    assert np.isfinite(d).all()
    return {name:np.quantile(d[:,i],[.025,.975]).tolist() for i,name in enumerate(['recall_diff_pp','fpr_diff_pp'])}


def screen(candidate,reference,train_costs,target):
    matched=bool(all(abs(x-target)<=.1 for x in train_costs))
    r=candidate['recall_pct']-reference['recall_pct'];f=candidate['fpr_pct']-reference['fpr_pct']
    return dict(train_costs_matched=matched,recall_diff_pp=r,fpr_diff_pp=f,
                exploratory_pass=bool(matched and r>=0 and f<=0 and (r>0 or f<0)))


def diagnose(pred,actual,sd,margin):
    bad=actual>4;plain=pred>4;flag=pred+margin>4
    c=counts(flag,bad);r=rates(c)
    added=flag&~plain
    r.update(added_true=int((added&bad).sum()),added_false=int((added&~bad).sum()),
             added_precision_pct=float(100*(added&bad).sum()/max(1,added.sum())))
    error=pred-actual
    correlation=lambda values:float(pd.Series(sd.ravel()).corr(pd.Series(values.ravel()),method='spearman'))
    if np.std(sd)>0:
        r.update(sd_absolute_error_spearman=correlation(abs(error)),
                 sd_underprediction_spearman=correlation(np.maximum(-error,0)))
    return r,c


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    p=json.loads((HERE/'protocol.json').read_text())
    assert sklearn.__version__=='1.7.2'
    candidate_path=Path(p['candidate']);prepared=Path(p['prepared']);model=Path(p['transition_model'])
    with candidate_path.open('rb') as f:candidate=pickle.load(f)
    cm=candidate['models'];kappa=float(candidate['kappa'])
    assert len(cm)==5 and kappa>0
    build_manifest=json.loads((candidate_path.parent/'manifest.json').read_text())
    assert sha(candidate_path)==build_manifest['model_sha256']
    splits,days=load_selected(prepared/'data.pkl')
    assert len(splits['train'])==188 and len(splits['val'])==40
    dyn=Dynamics(model);original=[m[0] for m in dyn.models]
    paths=[candidate_path,candidate_path.parent/'manifest.json',prepared/'data.pkl',model,HERE/'protocol.json',Path(__file__)]
    paths+=list((EXP/'e30').glob('*.py'))
    hashes={str(path):sha(path) for path in paths}
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  input_and_source_sha256=hashes,protocol=p,kappa_candidate=kappa,python=sys.executable,sklearn=sklearn.__version__,
                  candidate_build_commit=build_manifest['code_commit'],test_evaluated=False,weight_updates=0)
    save(a.out/'manifest.json',manifest)
    result=dict(splits={},kappa_candidate=kappa,test_evaluated=False,weight_updates=0)
    all_counts={}
    start=time.monotonic()
    for split in ['train','val']:
        env=BatchEnv(days,dyn);episodes=all_episodes(splits[split],5);env.reset(episodes)
        states=[];s0=[];s1=[];actual=[]
        for day,_ in episodes:actual.append(days[day]['state'][1:,0])
        actual=np.stack(actual).reshape(-1,5,95)
        for t in range(95):
            action=env.historical_action();air=action_to_air(action)
            zone=air[:,None]*env.x['frac'][:,t]
            x=np.column_stack([env.state,env.x['exo'][:,t],zone/1000.,env.prev_zone/1000.])
            s0.append(np.stack([m.predict(x) for m in original]).std(0))
            s1.append(np.stack([m.predict(x) for m in cm]).std(0))
            _,_,_,done,_=env.step(action)
            states.append(env.state[:,0].copy())
        assert done
        pred=np.stack(states,1).reshape(-1,5,95)
        sd0=np.stack(s0,1).reshape(-1,5,95);sd1=np.stack(s1,1).reshape(-1,5,95)
        # 같은 고정 전이를 재사용한 e53 감사의 기록 재생과 일치해야 한다.
        ref=Path('/home/jio/fcsrl_src/results/e53_margin_audit_20261112/calibration')/f'{split}_recorded_replay.npz'
        with np.load(ref) as old:
            np.testing.assert_array_equal(pred.reshape(-1,95),old['nh4'])
            np.testing.assert_array_equal(sd0.reshape(-1,95),old['sd'])
        flags=[(np.zeros_like(pred),0),(np.zeros_like(pred),.45),(sd0,10*sd0),(sd1,kappa*sd1)]
        result['splits'][split]={}
        all_counts[split]={}
        for name,(sd,margin) in zip(NAMES,flags):
            z,c=diagnose(pred,actual,sd,margin)
            result['splits'][split][name]=z;all_counts[split][name]=c
        result['splits'][split]['observed_cost']=float((actual>4).sum(2).mean())
        result['splits'][split]['sd_means']=dict(original=float(sd0.mean()),candidate=float(sd1.mean()))
        # 진단용 시간 구간은 사전에 정한 0~6h와 6~23.75h이며 선택에 쓰지 않는다.
        result['splits'][split]['time_bins']={}
        for label,sl in [('early_0_6h',slice(0,24)),('late_6_23_75h',slice(24,95))]:
            result['splits'][split]['time_bins'][label]={name:diagnose(pred[:,:,sl],actual[:,:,sl],sd[:,:,sl] if isinstance(sd,np.ndarray) else sd,
                         margin[:,:,sl] if isinstance(margin,np.ndarray) else margin)[0] for name,(sd,margin) in zip(NAMES,flags)}
        xs=[];current=[]
        for day in splits[split]:
            z=days[day];previous=np.concatenate([z['zone'][:1],z['zone'][:-2]])
            xs.append(np.column_stack([z['state'][:-1],z['exo'][:-1],z['zone'][:-1]/1000.,previous/1000.]))
            current.append(z['state'][:-1,0])
        x=np.concatenate(xs);current=np.concatenate(current)
        deltas=np.stack([m.predict(x) for m in original])
        one=np.clip(current[None,:]+deltas,0,dyn.hi[0]).reshape(5,len(splits[split]),95).transpose(1,0,2)
        osd=np.repeat(deltas.std(0).reshape(-1,1,95),5,axis=1)
        nsd=np.repeat(np.stack([m.predict(x) for m in cm]).std(0).reshape(-1,1,95),5,axis=1)
        result['splits'][split]['observed_state_one_step']={
            'original':diagnose(one,actual,osd,10*osd)[0],
            'candidate':diagnose(one,actual,nsd,kappa*nsd)[0]}
        np.savez_compressed(a.out/f'{split}_signals.npz',pred=pred,actual=actual,original_sd=sd0,candidate_sd=sd1,days=np.array(splits[split]))
        print(split,{name:result['splits'][split][name] for name in NAMES},flush=True)
    val=result['splits']['val'];train=result['splits']['train']
    result['decision']=screen(val['candidate'],val['original'],[train['candidate']['cost'],train['original']['cost']],train['observed_cost'])
    ci=block_intervals(all_counts['val']['candidate'],all_counts['val']['original'])
    result['decision']['paired_block_ci95']=ci
    result['decision']['reliability_flag']=bool(result['decision']['exploratory_pass'] and ci['recall_diff_pp'][0]>=0 and ci['fpr_diff_pp'][1]<=0)
    result['elapsed_seconds']=time.monotonic()-start
    assert all(sha(path)==expected for path,expected in hashes.items())
    result['hashes_unchanged']=True
    save(a.out/'results.json',result)
    print('DECISION',result['decision'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
