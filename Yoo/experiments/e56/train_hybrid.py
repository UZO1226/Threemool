"""e56 대기 작업용 실행기. CPU1 수집, PPO CPU1 또는 FCSRL GPU1 갱신."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm

import train as base
import hybrid as h


def run(a):
    manifest=json.loads(a.manifest.read_text())
    assert all(base.sha(p)==v for p,v in manifest['hashes'].items())
    if a.resume:
        assert (a.out/'resume.pt').exists()
    else:a.out.mkdir(parents=True,exist_ok=False)
    device='cuda:0' if a.method=='fcsrl' else 'cpu'
    c=base.common.config(a.updates)
    learner,collector=h.make_pair(a.method,a.seed,a.prepared,c,a.prior_mode,device)
    rng=np.random.default_rng(a.seed)
    metadata=dict(method=a.method,seed=a.seed,prior_mode=a.prior_mode,backend=device,updates=a.updates,
                  code_commit=manifest['code_commit'],data_sha256=base.sha(a.prepared/'data.pkl'),
                  model_sha256=base.sha(a.prepared/'surrogate.pkl'))
    records=h.restore(a.out/'resume.pt',learner,rng,metadata) if a.resume else []
    base.save(a.out/'effective_config.json',dict(**metadata,config=c.to_dict(),cpu_threads=1,
        gpu_physical_index=1 if device!='cpu' else None,collector='CPU shadow',permutation_rng='CPU',
        cost_limit=12,observation_dim=11,test_evaluated=False,prepared=str(a.prepared)))
    splits,days=base.load_selected(a.prepared/'data.pkl')
    env=base.BatchEnv(days,base.Dynamics(a.prepared/'surrogate.pkl'))
    original_prior={k:v.detach().cpu().clone() for k,v in learner.prior.state_dict().items()}
    norm=learner.obs_normalizer.state();start=time.monotonic();initial_updates=len(records)
    for u in tqdm(range(initial_updates+1,c.n_updates+1),desc=f'e56 {a.method} hybrid s{a.seed}',mininterval=10):
        tick=time.monotonic();h.sync_collector(learner,collector);h.configure('cpu')
        buf,diag=base.e30.collect(collector,c,env,splits['train'],rng)
        collection=time.monotonic()-tick
        tick=time.monotonic();info=h.update(learner,buf);updating=time.monotonic()-tick
        learner.pid.update(diag['daily_cost_mean'],c.cost_limit);learner.lagrg=learner.pid.get_lagrg()
        records.append(dict(update=u,**{'lambda':learner.lagrg},**diag,**info,
                            collection_seconds=collection,update_seconds=updating,cpu_threads=torch.get_num_threads()))
        h.checkpoint(a.out/'resume.pt',learner,rng,records,metadata)
        base.save(a.out/'train_log.json',records)
        elapsed=time.monotonic()-start
        base.save(a.out/'progress.json',dict(phase='training',update=u,updates=c.n_updates,steps=u*6080,
            total_steps=c.actual_steps,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/(u-initial_updates)*(c.n_updates-u),
            backend=device,cpu_threads=1,collection_seconds=collection,update_seconds=updating))
    assert learner.obs_normalizer.state()==norm
    assert all(torch.equal(v.detach().cpu(),original_prior[k]) for k,v in learner.prior.state_dict().items())
    h.sync_collector(learner,collector);h.configure('cpu')
    collector.save(a.out/'model_final')
    base.save(a.out/'model_final/prior_mode.json',dict(prior_mode=a.prior_mode,zero_output_saved=a.prior_mode=='off'))
    restored=base.common.make_agent(a.method,a.seed,a.prepared,c);restored.load(a.out/'model_final')
    assert base.fingerprint(restored)==base.fingerprint(collector)
    summaries={}
    for split in ['val','train']:
        base.save(a.out/'progress.json',dict(phase='evaluating_'+split,steps=c.actual_steps,backend=device))
        summaries[split+'_final']=base.evaluate(collector,env,splits[split],a.out,split)
    base.save(a.out/'metrics.json',dict(method=a.method,seed=a.seed,prior_mode=a.prior_mode,**summaries,
        steps=c.actual_steps,normalizer_and_prior_frozen=True,test_evaluated=False,backend=device,cpu_threads=1,
        elapsed_seconds=time.monotonic()-start,resumed_from_update=initial_updates))
    assert all(base.sha(p)==v for p,v in manifest['hashes'].items())
    base.save(a.out/'progress.json',dict(phase='complete',steps=c.actual_steps,backend=device,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ppo','fcsrl'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--prior-mode',choices=['on','off'],default='off')
    p.add_argument('--updates',type=int,default=66)
    p.add_argument('--prepared',type=Path,default=Path('/home/jio/fcsrl_src/results/e50/prepared'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--resume',action='store_true');run(p.parse_args())
