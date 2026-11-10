"""분리 비용/PID, 원본 표현 갱신, Actor 보정과 저장/복원을 검증한다."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from dual import config, make_agent, base_agent, combine_advantage, LIMITS, METHODS, save
from train import collect, evaluate
from darrow_env import BatchEnv,Dynamics,load_days
from fcsrl.utils import DeviceConfig


def main(prepared,out):
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    data = pd.read_pickle(prepared/'data.pkl')
    costs=[]
    for day in data['days']['train']:
        g=data['data'].loc[day]
        costs.append([(g.nh4.to_numpy()[1:]>4).sum(),(g[['do1','do2','do3']].to_numpy()[1:].max(1)>2).sum()])
    assert np.array_equal(np.ceil(np.mean(costs,0)),LIMITS)
    splits,days=load_days(prepared/'data.pkl')
    env=BatchEnv(days,Dynamics(prepared/'surrogate.pkl'))
    c=replace(config(1),n_env=2,horizon=190,total_steps=380,update_epochs=2)
    base=base_agent('fcsrl',0,prepared,c)
    dual=make_agent('fcsrl',0,prepared,c)
    sample=env.reset([(d,0) for d in splits['train'][:3]])
    assert np.array_equal(base.act_det(sample),dual.act_det(sample))
    with torch.no_grad():
        on=base._t(base.obs_normalizer(sample))
        assert torch.allclose(base.values(on)[1],dual.values(on)[1].sum(-1),atol=1e-6)
    b,diag=collect(dual,c,env,splits['train'],np.random.default_rng(0))
    assert b['end'].sum()==4 and b['terminated'].sum()==0
    assert np.array_equal(b['cost'],b['cost_parts'].sum(-1))
    with torch.no_grad():
        lp=dual.dist(dual._t(b['obs'])).log_prob(dual._t(b['act'])).sum(-1).numpy()
    assert np.allclose(lp,b['logp'],atol=1e-5)
    # Independent multipliers do not let a safe DO average cancel NH4 excess.
    probe=make_agent('ppo',0,prepared,c)
    probe.update_constraints([9.,0.])
    assert probe.multipliers[0]>0 and probe.multipliers[1]==0
    ar=np.array([1.,2.,4.]);ac=np.array([[1.,3.],[2.,4.],[4.,2.]])
    expected=((ar-ar.mean())/(ar.std()+1e-8)-2*(ac[:,0]-ac[:,0].mean())-3*(ac[:,1]-ac[:,1].mean()))/6
    assert np.allclose(combine_advantage(ar,ac,[2.,3.]),expected)
    flat=dict(b,adv_c=b['adv_c'].sum(-1),ret_c=b['ret_c'].sum(-1))
    rng=torch.get_rng_state().clone()
    base.update(flat)
    expected_rng=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    info=dual.update(b)
    # At initial zero multipliers, actor, std, original representation and RNG stay exact.
    for name in ('pi','encoder','fixed_encoder','feasi_head','proj_layer','post_proj','prior'):
        assert all(torch.equal(v,getattr(base,name).state_dict()[k]) for k,v in getattr(dual,name).state_dict().items()),name
    assert torch.equal(base.log_std,dual.log_std) and torch.equal(torch.get_rng_state(),expected_rng)
    variants={}
    for method in METHODS:
        agent=make_agent(method,0,prepared,c)
        torch.set_rng_state(rng)
        metrics=agent.update(b)
        assert all(np.isfinite(v) for v in metrics.values())
        if method=='fcsrl_actor':
            for name in ('encoder','fixed_encoder','vr','vc'):
                assert all(torch.equal(v,getattr(dual,name).state_dict()[k]) for k,v in getattr(agent,name).state_dict().items()),name
        with tempfile.TemporaryDirectory() as path:
            agent.save(path)
            restored=make_agent(method,8,prepared,c)
            restored.load(path)
            assert np.array_equal(agent.act_det(sample),restored.act_det(sample))
        variants[method]=metrics
    # Train-only feasibility witnesses; these are not validation-selected policies.
    witnesses=[]
    for air in (3000.,4000.,5000.,6000.):
        sm,_=evaluate(env,splits['train'],lambda x:np.full(len(x),air/9000*2-1))
        witnesses.append({'constant_air_m3_h':air,**sm})
    save(out,{'passed':True,'train_observed_mean_costs':np.mean(costs,0).tolist(),
         'limits':LIMITS.tolist(),'independent_pid_verified':True,'cost_decomposition_exact':True,
         'initial_actor_encoder_ema_rng_exact':True,'actor_only_head_isolation':True,
         'raw_logp_verified':True,'checkpoint_restore_exact':True,'variants':variants,
         'train_only_constant_action_witnesses':witnesses})
    print('PASS',json.dumps(witnesses),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.prepared,a.out)
