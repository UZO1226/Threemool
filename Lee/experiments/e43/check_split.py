"""Check target identity, original objective, gradients, EMA and serialization."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from split_feasibility import TwoHeadFCSRL
from ppo_fcsrl import PPOFCSRLAgent,RepresentationConfig
from ppo_config import PPOConfig

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
c=PPOConfig(total_steps=6080,n_env=8,horizon=760,n_chunks=1,cost_limit=12)
r=RepresentationConfig()


def build(mode=None,seed=0):
    torch.manual_seed(seed)
    return (PPOFCSRLAgent(11,1,config=c,repr_config=r) if mode is None else
        TwoHeadFCSRL(11,1,config=c,repr_config=r,mode=mode))


base=build();base_rng=torch.random.get_rng_state().clone()
split=build('split');torch.testing.assert_close(torch.random.get_rng_state(),base_rng,rtol=0,atol=0)
control=build('capacity');torch.testing.assert_close(torch.random.get_rng_state(),base_rng,rtol=0,atol=0)
names=('pi','vr','vc','encoder','fixed_encoder','feasi_head','proj_layer','post_proj')
for name in names:
    for key,value in getattr(base,name).state_dict().items():
        torch.testing.assert_close(value,getattr(split,name).state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(value,getattr(control,name).state_dict()[key],rtol=0,atol=0)
for key,value in split.second_head.state_dict().items():torch.testing.assert_close(value,control.second_head.state_dict()[key],rtol=0,atol=0)
assert any(not torch.equal(x,y) for x,y in zip(split.feasi_head.parameters(),split.second_head.parameters()))

buf={'part_cost':np.zeros((3,1,2),np.float32),'part_next':np.zeros((3,1,2),np.float32),
    'end':np.array([[0],[1],[0]],np.float32),'terminated':np.array([[0],[0],[1]],np.float32)}
nh4=np.array([1,0,0],np.float32);do=np.array([0,0,1],np.float32)
for t in range(3):split.collect_aux(buf,t,np.zeros((1,11)),np.array([nh4[t]+do[t]]),{'v_nh4':nh4[t:t+1],'v_do':do[t:t+1]})
np.testing.assert_array_equal(buf['part_cost'][:,0],np.column_stack([nh4,do]))
buf['part_next'][:]=np.array([[[.4,.8]],[[.2,.4]],[[.6,.8]]]);split.finish_buffer(buf)
np.testing.assert_allclose(buf['feasi_parts'][:,0],[[1,.324],[.18,.36],[0,1]],rtol=1e-6)
for t in range(3):control.collect_aux(buf,t,np.zeros((1,11)),np.array([nh4[t]+do[t]]),{'v_nh4':nh4[t:t+1],'v_do':do[t:t+1]})
np.testing.assert_array_equal(buf['part_cost'][:,0],np.repeat((nh4+do)[:,None],2,axis=1))

# The redundant-head limit must recover the actual original imported loss
# and common-module gradients, not another copy of the new implementation.
control=build('capacity');control.second_head.load_state_dict(control.feasi_head.state_dict())
base=build();rng=np.random.default_rng(43);n=32
batch={'obs':rng.normal(size=(n,11)).astype(np.float32),'obs_next':rng.normal(size=(n,11)).astype(np.float32),
    'act_exec':rng.uniform(-1,1,(n,1)).astype(np.float32),'feasi':rng.uniform(0,2,n).astype(np.float32)}
batch['feasi_parts']=np.repeat(batch['feasi'][:,None],2,axis=1);idx=torch.arange(n)
base.encoder_optim.step=lambda:None;control.encoder_optim.step=lambda:None
old_metrics=base.update_aux(batch,idx);new_metrics=control.update_aux(batch,idx)
for key in ('encoder_dynamics','encoder_pred_feasi','encoder_pred_feasi_MSE','encoder'):
    np.testing.assert_allclose(old_metrics[key],new_metrics[key],rtol=1e-6,atol=1e-7)
for name in ('encoder','proj_layer','post_proj'):
    for left,right in zip(getattr(base,name).parameters(),getattr(control,name).parameters()):
        torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-6)

split=build('split');batch['feasi_parts']=np.column_stack([np.linspace(0,1,n),np.linspace(1,0,n)]).astype(np.float32)
before={key:value.clone() for key,value in split.fixed_encoder.state_dict().items()}
metrics=split.update_aux(batch,idx)
assert all(np.isfinite(value) for value in metrics.values())
for module in (split.encoder,split.feasi_head,split.second_head):
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in module.parameters())
for key,value in before.items():torch.testing.assert_close(value,split.fixed_encoder.state_dict()[key],rtol=0,atol=0)
split.finish_update()
for key,value in split.fixed_encoder.state_dict().items():
    torch.testing.assert_close(value,before[key]*(1-r.tau)+split.encoder.state_dict()[key]*r.tau)
if a.checkpoint.exists():raise FileExistsError(a.checkpoint)
split.save(a.checkpoint);loaded=build('split',seed=4);loaded.load(a.checkpoint)
np.testing.assert_array_equal(split.act_det(batch['obs']),loaded.act_det(batch['obs']))
np.testing.assert_array_equal(split.next_parts(batch['obs']),loaded.next_parts(batch['obs']))
result={'common_parent_parameters_exact':True,'common_parent_rng_preserved':True,
    'split_capacity_initial_parameters_exact':True,'second_head_independently_initialized':True,
    'hand_computed_reset_termination_targets_passed':True,'component_vs_sum_labels_passed':True,
    'redundant_head_original_objective_and_common_gradients_passed':True,'nonzero_finite_gradients':True,
    'ema_passed':True,'checkpoint_round_trip_exact':True,'test_evaluated':False,'policy_performance_tested':False}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
