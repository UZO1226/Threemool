"""학습 날짜만 사용해 출력 보정과 원본 FCSRL 갱신 보존을 점검한다."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from preserve import make_agent
from behavior import observations
from env import AerationEnv
from fcsrl.utils import DeviceConfig
from ppo_config import PPOConfig
from ppo_lag import gae
from ppo_fcsrl import feasibility_targets
from evaluate import batched_rollout, run_policy, summarize_rows

p=argparse.ArgumentParser()
for name in ('data','model','prepared','baseline','out'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
DeviceConfig().select_device(-1);torch.set_num_threads(1)
c=PPOConfig(total_steps=256,n_env=1,horizon=256,n_chunks=1,update_epochs=2,init_log_std=-1.6)
base=make_agent('fcsrl',0,a.prepared,c)
agent=make_agent('fcsrl_preserve',0,a.prepared,c)
env=AerationEnv(a.data,a.model,split='train',seed=0)
sample=np.concatenate([observations(env.day_data[d]) for d in sorted(env.days)[:3]])
assert np.array_equal(base.act_det(sample),agent.act_det(sample))
obs,_=env.reset()
shapes=[('obs',(11,)),('obs_next',(11,)),('act',(1,)),('act_exec',(1,)),('logp',()),('rew',()),('cost',()),('vr',()),('vc',()),('nvr',()),('nvc',()),('end',()),('terminated',()),('next_feasi',())]
b={k:np.zeros((256,1)+sh,np.float32) for k,sh in shapes}
for t in range(256):
    on=base.obs_normalizer(obs[None]);ot=base._t(on)
    with torch.no_grad():
        dist=base.dist(ot);act=dist.sample();vr,vc=base.values(ot)
        b['logp'][t]=dist.log_prob(act).sum(-1).numpy()
        b['vr'][t],b['vc'][t]=vr.numpy(),vc.numpy()
    b['obs'][t],b['act'][t]=on,act.numpy();b['act_exec'][t]=np.clip(act.numpy(),-1,1)
    obs,rew,term,trunc,info=env.step(b['act_exec'][t,0]);assert not term and not trunc
    b['rew'][t],b['cost'][t]=rew,info['cost']
    no=base.obs_normalizer(obs[None]);b['obs_next'][t]=no
    with torch.no_grad():
        vr,vc=base.values(base._t(no));b['nvr'][t],b['nvc'][t]=vr.numpy(),vc.numpy()
        b['next_feasi'][t]=base.next_feasibility(no)
b['adv_r'],b['ret_r']=gae(b['rew'],b['vr'],b['nvr'],b['end'])
b['adv_c'],b['ret_c']=gae(b['cost'],b['vc'],b['nvc'],b['end'])
b['feasi']=feasibility_targets(b['cost'],b['next_feasi'],b['end'],b['terminated'])
flat={k:v.reshape(256,*v.shape[2:]) for k,v in b.items()}
with torch.no_grad():
    assert np.allclose(base.dist(base._t(flat['obs'])).log_prob(base._t(flat['act'])).sum(-1).numpy(),flat['logp'],atol=1e-6)
rng=torch.get_rng_state().clone();base.update(flat);base_rng=torch.get_rng_state().clone()
torch.set_rng_state(rng);metrics=agent.update(flat)
assert torch.equal(base_rng,torch.get_rng_state())
for name in ('encoder','fixed_encoder','feasi_head','proj_layer','post_proj','prior'):
    assert all(torch.equal(v,getattr(base,name).state_dict()[k]) for k,v in getattr(agent,name).state_dict().items()),name
assert torch.equal(base.log_std,agent.log_std)
assert base.obs_normalizer.state()==agent.obs_normalizer.state()
assert all(np.isfinite(v) for v in metrics.values())
with tempfile.TemporaryDirectory() as td:
    agent.save(td);restored=make_agent('fcsrl_preserve',8,a.prepared,c);restored.load(td)
    assert np.array_equal(agent.act_det(sample),restored.act_det(sample))
# 실제 학습된 checkpoint에서 train 관측만 사용한 출력 보존 검사.
trained=make_agent('fcsrl_preserve',0,a.prepared,PPOConfig(total_steps=200000,n_chunks=2,init_log_std=-1.6))
trained.load(a.baseline)
normalized=trained.obs_normalizer(sample)
ids=np.random.default_rng(25000).permutation(len(sample))
trained._preserve_obs=trained._t(normalized[ids]);trained._fit_size=512
prior=deepcopy(trained.prior.state_dict());online=deepcopy(trained.encoder.state_dict())
trained_metrics=trained.finish_update()
assert trained_metrics['preserve_accepted']==1
assert trained_metrics['preserve_after_error'] <= .8*trained_metrics['preserve_before_error']
assert trained_metrics['preserve_after_kl'] <= trained_metrics['preserve_before_kl']*1.01+1e-10
assert all(torch.equal(v,trained.prior.state_dict()[k]) for k,v in prior.items())
assert all(torch.equal(v,trained.encoder.state_dict()[k]) for k,v in online.items())
# 표현이 변하지 않으면 보정이 기존 정책을 바꾸지 않는다.
with torch.no_grad():
    features=trained.policy_input(trained._preserve_obs).detach()
    teachers=[m(features).detach() for m in (trained.pi,trained.vr,trained.vc)]
before=[deepcopy(m.state_dict()) for m in (trained.pi,trained.vr,trained.vc)]
identity=trained.compensate(features,teachers,[trained.log_std.exp().detach(),torch.tensor(1.),torch.tensor(1.)],512)
assert identity['preserve_accepted']==0 and identity['preserve_before_error']==0
assert all(torch.equal(v,m.state_dict()[k]) for m,state in zip((trained.pi,trained.vr,trained.vc),before) for k,v in state.items())
eps=[(sorted(env.days)[0],0),(sorted(env.days)[1],1)]
seq=AerationEnv(a.data,a.model,split='train',fixed_episodes=eps)
scalar=summarize_rows(run_policy(seq,lambda o,e:trained.act_det(o)[0],2),15.)
bat=AerationEnv(a.data,a.model,split='train',fixed_episodes=eps)
vector=batched_rollout(bat,bat.sur,lambda o:trained.act_det(o)[:,0],15.)[0]
assert abs(scalar['air_m3']-vector['air_m3']) < .1 and scalar['cost']==vector['cost']
report={'passed':True,'initial_policy_equal':True,'original_encoder_and_ema_update_exact':True,'rng_stream_unchanged':True,'prior_norm_std_unchanged':True,'raw_log_probability':True,'checkpoint_restore_exact':True,'identity_compensation_exact':True,'trained_checkpoint_train_states_only':True,'scalar_batched_air_difference':scalar['air_m3']-vector['air_m3'],'synthetic_update':metrics,'trained_sync':trained_metrics}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
