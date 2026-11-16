"""Check daily boundaries, unchanged PPO inputs and inherited objective."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from finite_day import FiniteDayFCSRL
from ppo_fcsrl import PPOFCSRLAgent,RepresentationConfig,feasibility_targets
from ppo_config import PPOConfig

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1)
c=PPOConfig(total_steps=6080,n_env=8,horizon=760,n_chunks=1,cost_limit=12);r=RepresentationConfig()
def build(candidate=True,seed=0):
    torch.manual_seed(seed)
    cls=FiniteDayFCSRL if candidate else PPOFCSRLAgent
    return cls(11,1,config=c,repr_config=r)
base=build(False);rng_after=torch.random.get_rng_state().clone();candidate=build()
torch.testing.assert_close(rng_after,torch.random.get_rng_state(),rtol=0,atol=0)
names=('pi','vr','vc','encoder','fixed_encoder','feasi_head','proj_layer','post_proj')
for name in names:
    for key,value in getattr(base,name).state_dict().items():
        torch.testing.assert_close(value,getattr(candidate,name).state_dict()[key],rtol=0,atol=0)
buf={'cost':np.array([[0,1],[0,0],[2,0]],np.float32),
    'next_feasi':np.array([[.4,.8],[.6,.4],[.3,.7]],np.float32),
    'end':np.array([[0,0],[1,1],[0,1]],np.float32),'terminated':np.zeros((3,2),np.float32)}
before={k:v.copy() for k,v in buf.items()};candidate.finish_buffer(buf)
np.testing.assert_allclose(buf['feasi'],[[0,1],[0,0],[2,0]],rtol=0,atol=0)
for key,value in before.items():np.testing.assert_array_equal(value,buf[key])
continuation=feasibility_targets(before['cost'],before['next_feasi'],before['end'],before['terminated'],.9)
np.testing.assert_allclose(continuation,[[.486,1],[.54,.36],[2,.63]],rtol=1e-6)
buf['end'][:]=0
expected=feasibility_targets(buf['cost'],buf['next_feasi'],buf['end'],buf['terminated'],.9)
candidate.finish_buffer(buf);np.testing.assert_array_equal(buf['feasi'],expected)
buf['terminated'][-1]=1;candidate.finish_buffer(buf)
np.testing.assert_array_equal(buf['feasi'][-1],buf['cost'][-1])

random=np.random.default_rng(48);n=32
batch={'obs':random.normal(size=(n,11)).astype(np.float32),'obs_next':random.normal(size=(n,11)).astype(np.float32),
    'act_exec':random.uniform(-1,1,(n,1)).astype(np.float32),'feasi':random.uniform(0,2,n).astype(np.float32)}
base=build(False);candidate=build()
base.encoder_optim.step=lambda:None;candidate.encoder_optim.step=lambda:None
left=base.update_aux(batch,torch.arange(n));right=candidate.update_aux(batch,torch.arange(n))
assert left==right
for name in ('encoder','feasi_head','proj_layer','post_proj'):
    for x,y in zip(getattr(base,name).parameters(),getattr(candidate,name).parameters()):
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
candidate=build();metrics=candidate.update_aux(batch,torch.arange(n))
assert all(np.isfinite(v) for v in metrics.values())
assert any(p.grad is not None and p.grad.abs().max()>0 for p in candidate.encoder.parameters())
if a.checkpoint.exists():raise FileExistsError(a.checkpoint)
candidate.save(a.checkpoint);loaded=build(seed=3);loaded.load(a.checkpoint)
np.testing.assert_array_equal(candidate.act_det(batch['obs']),loaded.act_det(batch['obs']))
np.testing.assert_array_equal(candidate.next_feasibility(batch['obs']),loaded.next_feasibility(batch['obs']))
out={'initial_parameters_rng_exact':True,'hand_computed_daily_targets_pass':True,'no_end_original_targets_exact':True,
    'true_terminal_targets_pass':True,'ppo_buffer_fields_unchanged':True,'inherited_objective_gradients_exact':True,
    'finite_nonzero_gradients':True,'checkpoint_roundtrip_exact':True,'policy_performance_tested':False,'test_evaluated':False}
a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
