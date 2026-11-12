"""Check severity targets, inherited objective and terminal adapter semantics."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from severity_agent import SeverityFCSRL,SeverityInfoEnv,severity_cost
from ppo_fcsrl import PPOFCSRLAgent,RepresentationConfig,feasibility_targets
from ppo_config import PPOConfig
from quantile_env import FrozenQuantileEnv

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1)
c=PPOConfig(total_steps=6080,n_env=8,horizon=760,n_chunks=1,cost_limit=12)
r=RepresentationConfig()
def build(candidate=True,weight=1.,seed=0):
    torch.manual_seed(seed)
    if candidate:return SeverityFCSRL(11,1,config=c,repr_config=r,severity_weight=weight)
    return PPOFCSRLAgent(11,1,config=c,repr_config=r)

base=build(False);rng_after=torch.random.get_rng_state().clone();candidate=build()
torch.testing.assert_close(rng_after,torch.random.get_rng_state(),rtol=0,atol=0)
names=('pi','vr','vc','encoder','fixed_encoder','feasi_head','proj_layer','post_proj')
for name in names:
    for key,value in getattr(base,name).state_dict().items():
        torch.testing.assert_close(value,getattr(candidate,name).state_dict()[key],rtol=0,atol=0)
states=np.array([[3,0,1,1,1],[4,0,2,2,2],[6,0,1,1,1],[4,0,3,1,1],[8,0,3,1,1]],np.float32)
cost=np.array([0,0,1,1,2],np.float32)
np.testing.assert_array_equal(severity_cost(cost,states),[0,0,1.5,1.5,3.5])
np.testing.assert_array_equal(severity_cost(cost,states,0),cost)
np.testing.assert_array_equal(severity_cost(cost,states)==0,cost==0)

# An auto-reset observation differs deliberately from its outgoing terminal.
terminal=np.column_stack([states,np.zeros((5,6))]).astype(np.float32)
reset=np.full_like(terminal,-99);reward=np.arange(5.)
done=np.array([False,True,False,True,True]);info={'v_nh4':np.array([0,0,1,0,1]),'v_do':np.array([0,0,0,1,1])}
result=(reset,reward,cost,done,terminal,info)
with patch.object(FrozenQuantileEnv,'step',return_value=result):
    adapted=SeverityInfoEnv.__new__(SeverityInfoEnv).step(np.zeros(5))
for i in range(5):assert adapted[i] is result[i]
np.testing.assert_array_equal(adapted[5]['next_state_for_aux'],states)
adapted[5]['next_state_for_aux'][0,0]=100
np.testing.assert_array_equal(terminal[:,:5],states)
assert 'next_state_for_aux' not in info

buf={'severity_cost':np.array([[1.5],[0],[2]],np.float32),'next_feasi':np.array([[.4],[.2],[.6]],np.float32),
    'end':np.array([[0],[1],[0]],np.float32),'terminated':np.array([[0],[0],[1]],np.float32),
    'cost':np.array([[1],[0],[1]],np.float32)}
actual=buf['cost'].copy();candidate.finish_buffer(buf)
np.testing.assert_allclose(buf['feasi'][:,0],[1.5,.18,2],rtol=1e-6)
np.testing.assert_array_equal(buf['cost'],actual)
original=build(weight=0)
buf['severity_cost']=actual.copy();original.finish_buffer(buf)
np.testing.assert_array_equal(buf['feasi'],feasibility_targets(actual,buf['next_feasi'],buf['end'],buf['terminated'],.9))

rng=np.random.default_rng(45);n=32
batch={'obs':rng.normal(size=(n,11)).astype(np.float32),'obs_next':rng.normal(size=(n,11)).astype(np.float32),
    'act_exec':rng.uniform(-1,1,(n,1)).astype(np.float32),'feasi':rng.uniform(0,4,n).astype(np.float32)}
base=build(False);original=build(weight=0)
base.encoder_optim.step=lambda:None;original.encoder_optim.step=lambda:None
left=base.update_aux(batch,torch.arange(n));right=original.update_aux(batch,torch.arange(n))
assert left==right
for name in ('encoder','feasi_head','proj_layer','post_proj'):
    for x,y in zip(getattr(base,name).parameters(),getattr(original,name).parameters()):
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
candidate=build();metrics=candidate.update_aux(batch,torch.arange(n))
assert all(np.isfinite(v) for v in metrics.values())
for name in ('encoder','feasi_head'):
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in getattr(candidate,name).parameters())
if a.checkpoint.exists():raise FileExistsError(a.checkpoint)
candidate.save(a.checkpoint);loaded=build(seed=3);loaded.load(a.checkpoint)
np.testing.assert_array_equal(candidate.act_det(batch['obs']),loaded.act_det(batch['obs']))
np.testing.assert_array_equal(candidate.next_feasibility(batch['obs']),loaded.next_feasibility(batch['obs']))
checks={'initial_parameters_and_rng_exact':True,'safe_boundary_severe_labels_pass':True,
    'terminal_adapter_outputs_unchanged_under_mock':True,'terminal_not_reset_labels_pass':True,
    'hand_computed_reset_termination_targets_pass':True,'actual_buffer_cost_unchanged':True,
    'original_label_objective_and_gradients_exact':True,'nonzero_finite_gradients':True,
    'checkpoint_roundtrip_exact':True,'policy_training':False,'real_environment_equivalence_checked':False,
    'policy_performance_tested':False,'test_evaluated':False}
a.out.write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
