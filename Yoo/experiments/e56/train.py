"""e50과 같은 학습에서 고정 모방 prior의 출력만 0으로 만드는 대조."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import sklearn
import torch
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'e30'))
import common
from darrow_env import BatchEnv, Dynamics, all_episodes
from ppo_fcsrl import PPOFCSRLAgent
from fcsrl.utils import DeviceConfig
spec=importlib.util.spec_from_file_location('original_e30_train',HERE.parent/'e30/train.py')
e30=importlib.util.module_from_spec(spec);spec.loader.exec_module(e30)
sys.path.insert(0,str(HERE.parent/'e45'))
from evaluate import load_selected, metrics, fingerprint
save,sha=common.save,common.sha


def make_agent(method,seed,prepared,config,prior_mode):
    agent=common.make_agent(method,seed,prepared,config)
    if prior_mode=='off':
        # 난수와 다른 가중치를 보존한다. checkpoint의 prior.pt도 영 출력이므로 기존 loader로 재현된다.
        with torch.no_grad():
            agent.prior[-1].weight.zero_()
            agent.prior[-1].bias.zero_()
    elif prior_mode!='on':
        raise ValueError(prior_mode)
    return agent


def evaluate(agent,env,dates,out,split):
    before=fingerprint(agent)
    episodes=all_episodes(dates,5);obs=env.reset(episodes)
    states,airs=[],[]
    for _ in range(95):
        obs,reward,cost,done,info=env.step(agent.act_det(obs)[:,0])
        assert np.allclose(reward,-info['air']/9000.)
        assert np.array_equal(cost,info['v_nh4']+info['v_do'])
        states.append(env.state.copy());airs.append(info['air'].copy())
    assert done and fingerprint(agent)==before
    states=np.stack(states,1);airs=np.stack(airs,1)
    result,rows,members=metrics(states,airs,episodes)
    # helper의 6/6 표시는 e50의 합산 한도12와 다르므로 혼합하지 않는다.
    result.pop('both_mean_limits');result.pop('both_limits_episode_pct')
    n=(states[:,:,0]>4).sum(1);d=(states[:,:,2:].max(2)>2).sum(1)
    result.update(cost=float((n+d).mean()),mean_cost_within_12=bool((n+d).mean()<=12),
                  nh4_viol_pct=float((100*n/95).mean()),do_viol_pct=float((100*d/95).mean()),
                  days_within_limit_pct=float(100*np.mean(n+d<=12)),n_episodes=len(episodes))
    save(out/f'{split}_episodes.json',rows)
    np.savez_compressed(out/f'{split}_rollout.npz',states=states,air=airs)
    return result


def run(a):
    assert sklearn.__version__=='1.9.1'
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text()) if a.manifest else None
    if manifest:
        assert all(sha(p)==h for p,h in manifest['hashes'].items())
    torch.set_num_threads(1);DeviceConfig().select_device(-1)
    c=common.config(a.updates)
    agent=make_agent(a.method,a.seed,a.prepared,c,a.prior_mode)
    splits,days=load_selected(a.prepared/'data.pkl')
    env=BatchEnv(days,Dynamics(a.prepared/'surrogate.pkl'))
    rng=np.random.default_rng(a.seed)
    save(a.out/'effective_config.json',dict(method=a.method,seed=a.seed,prior_mode=a.prior_mode,
         config=c.to_dict(),prepared=str(a.prepared),observation_dim=11,cost_limit=12,
         initial_mean_without_prior='0 normalized action = 4500 m3/h',
         evaluation='last checkpoint, deterministic train/val',test_evaluated=False,
         code_commit=manifest['code_commit'] if manifest else None))
    norm=agent.obs_normalizer.state();prior={k:v.clone() for k,v in agent.prior.state_dict().items()}
    records=[];start=time.monotonic()
    for u in tqdm(range(1,c.n_updates+1),desc=f'e56 {a.method} prior={a.prior_mode} s{a.seed}',mininterval=10):
        buf,diag=e30.collect(agent,c,env,splits['train'],rng)
        update=agent.update(buf)
        assert all(np.isfinite(v) for v in update.values())
        modules=[agent.pi,agent.vr,agent.vc]
        if isinstance(agent,PPOFCSRLAgent):
            modules += [agent.encoder,agent.fixed_encoder,agent.feasi_head,agent.proj_layer,agent.post_proj]
        assert all(torch.isfinite(p).all() for m in modules for p in m.parameters())
        agent.pid.update(diag['daily_cost_mean'],c.cost_limit);agent.lagrg=agent.pid.get_lagrg()
        records.append(dict(update=u,**{'lambda':agent.lagrg},**diag,**update))
        elapsed=time.monotonic()-start
        save(a.out/'train_log.json',records)
        save(a.out/'progress.json',dict(phase='training',update=u,updates=c.n_updates,steps=u*6080,
             total_steps=c.actual_steps,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/u*(c.n_updates-u)))
    assert norm==agent.obs_normalizer.state()
    assert all(torch.equal(v,prior[k]) for k,v in agent.prior.state_dict().items())
    agent.save(a.out/'model_final')
    save(a.out/'model_final/prior_mode.json',dict(prior_mode=a.prior_mode,zero_output_saved=a.prior_mode=='off'))
    # 기존 loader로도 prior 비활성화가 유지되는지 확인한다.
    restored=common.make_agent(a.method,a.seed,a.prepared,c);restored.load(a.out/'model_final')
    assert fingerprint(restored)==fingerprint(agent)
    if not a.smoke:
        summaries={}
        for split in ['val','train']:
            save(a.out/'progress.json',dict(phase='evaluating_'+split,steps=c.actual_steps))
            summaries[split+'_final']=evaluate(agent,env,splits[split],a.out,split)
        save(a.out/'metrics.json',dict(method=a.method,seed=a.seed,prior_mode=a.prior_mode,**summaries,
             steps=c.actual_steps,normalizer_and_prior_frozen=True,test_evaluated=False,elapsed_seconds=time.monotonic()-start))
    if manifest:
        assert all(sha(p)==h for p,h in manifest['hashes'].items())
    save(a.out/'progress.json',dict(phase='complete',steps=c.actual_steps,smoke=a.smoke,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['ppo','fcsrl'],required=True)
    p.add_argument('--prior-mode',choices=['on','off'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=66)
    p.add_argument('--prepared',type=Path,default=Path('/home/jio/fcsrl_src/results/e50/prepared'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
