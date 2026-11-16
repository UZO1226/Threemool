"""CPU1 수집과 선택적인 GPU 갱신. 기존 FCSRL 손실과 CPU 난수열을 유지한다."""
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path

import numpy as np
import torch
import train as base

MODULES=['pi','vr','vc','prior','encoder','fixed_encoder','feasi_head','proj_layer','post_proj']
OPTIMIZERS=['opt_pi','opt_v','encoder_optim']


def configure(device):
    if device!='cpu':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1', 'Physical GPU1 only'
        assert torch.cuda.device_count()==1
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    base.DeviceConfig.DEVICE=torch.device(device)
    torch.set_num_threads(1)
    assert torch.get_num_threads()==1


def modules(agent):
    return {n:getattr(agent,n) for n in MODULES if hasattr(agent,n)}


def make_pair(method,seed,prepared,c,mode,device):
    configure('cpu')
    learner=base.make_agent(method,seed,prepared,c,mode)
    collector=copy.deepcopy(learner)
    if device!='cpu':
        configure(device);learner.dev=torch.device(device)
        for module in modules(learner).values():module.to(device)
        learner.log_std=torch.nn.Parameter(learner.log_std.detach().to(device))
        learner.opt_pi=torch.optim.Adam(list(learner.pi.parameters())+[learner.log_std],lr=c.actor_lr)
        learner.opt_v=torch.optim.Adam(list(learner.vr.parameters())+list(learner.vc.parameters()),lr=c.critic_lr)
        if hasattr(learner,'encoder_optim'):
            params=[p for m in (learner.encoder,learner.feasi_head,learner.proj_layer,learner.post_proj) for p in m.parameters()]
            learner.encoder_optim=torch.optim.Adam(params,lr=learner.repr_config.encoder_lr)
    return learner,collector


@torch.no_grad()
def sync_collector(learner,collector):
    for name,module in modules(learner).items():
        getattr(collector,name).load_state_dict({k:v.detach().cpu() for k,v in module.state_dict().items()})
    collector.log_std.copy_(learner.log_std.detach().cpu())
    collector.lagrg=learner.lagrg
    assert learner.obs_normalizer.state()==collector.obs_normalizer.state()


@contextmanager
def cpu_permutations():
    # 기존 update의 device별 randperm만 CPU RNG로 통일한다. 손실/순서/반복 수는 유지한다.
    original=torch.randperm
    def draw(n,device=None,**kwargs):
        return original(n,device='cpu',**kwargs).to(device or 'cpu')
    torch.randperm=draw
    try:yield
    finally:torch.randperm=original


@contextmanager
def gpu_slot(device):
    if device=='cpu':
        yield
        return
    with open('/home/jio/fcsrl_src/results/e56/gpu1_update.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def update(learner,buf):
    device=str(learner.dev)
    configure(device)
    with gpu_slot(device),cpu_permutations():
        info=learner.update(buf)
        if device!='cpu':torch.cuda.synchronize()
    assert all(np.isfinite(v) for v in info.values())
    assert all(torch.isfinite(p).all() for module in modules(learner).values() for p in module.parameters())
    return info


def checkpoint(path,agent,rng,records,metadata):
    payload=dict(modules={n:{k:v.detach().cpu() for k,v in m.state_dict().items()} for n,m in modules(agent).items()},
        log_std=agent.log_std.detach().cpu(),optimizers={n:getattr(agent,n).state_dict() for n in OPTIMIZERS if hasattr(agent,n)},
        pid=vars(agent.pid).copy(),lagrg=agent.lagrg,normalizer=agent.obs_normalizer.state(),
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),sampler_rng=rng.bit_generator.state,
        records=records,metadata=metadata)
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def restore(path,agent,rng,expected):
    z=torch.load(path,map_location=agent.dev,weights_only=False)
    assert z['metadata']==expected,(z['metadata'],expected)
    for n,s in z['modules'].items():getattr(agent,n).load_state_dict(s)
    with torch.no_grad():agent.log_std.copy_(z['log_std'])
    for n,s in z['optimizers'].items():getattr(agent,n).load_state_dict(s)
    vars(agent.pid).update(z['pid']);agent.lagrg=z['lagrg']
    agent.obs_normalizer.load(z['normalizer'])
    torch.set_rng_state(z['torch_rng'].cpu());np.random.set_state(z['numpy_rng']);rng.bit_generator.state=z['sampler_rng']
    return z['records']
