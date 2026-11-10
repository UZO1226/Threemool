"""전체 수집/학습/복원 경로와 입력 정보 경계를 본 실행 전에 검사한다."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from command import (HERE, METHODS, MODES, CommandEnv, Dynamics, load_days, make_agent,
                     config, save, sha, BatchEnv, PPOFCSRLAgent)
from train import collect, evaluate
from fcsrl.utils import DeviceConfig


def modules(agent):
    return [agent.pi, agent.vr, agent.vc] + ([agent.encoder, agent.fixed_encoder, agent.feasi_head,
            agent.proj_layer, agent.post_proj] if isinstance(agent, PPOFCSRLAgent) else [])


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    splits, days = load_days(a.prepared / 'data.pkl')
    assert [len(splits[k]) for k in ('train', 'val', 'test')] == [188, 40, 41]
    dyn = Dynamics(a.corrected)
    eps = [(splits['train'][0], 0)]
    env = CommandEnv(days, dyn, 'previous')
    original = BatchEnv(days, dyn)
    obs, old = env.reset(eps), original.reset(eps)
    assert np.array_equal(obs[:, :11], old) and np.all(obs[:, 11:] == 0)
    obs, *_ = env.step(np.array([0.]))
    old, *_ = original.step(np.array([0.]))
    assert np.array_equal(obs[:, :11], old)
    assert np.allclose(obs[:, 11:] * 9000, env.prev_zone)
    assert np.allclose(obs[:, 11:].sum(1), 0.5)
    before = env.obs().copy()
    env.x['state'][:, env.t + 1:] = 999
    env.x['frac'][:, env.t:] = 0
    env.x['air'][:, env.t:] = 999
    assert np.array_equal(before, env.obs())
    assert np.all(env.reset(eps)[:, 11:] == 0)
    records = []
    for method in METHODS:
        control = None
        initial_weights = None
        for mode in MODES:
            start = time.monotonic()
            c = config(1)
            agent = make_agent(method, 0, a.prepared, c)
            init = [{k: v.clone() for k,v in m.state_dict().items()} for m in modules(agent)]
            if initial_weights is None:
                initial_weights = init
            else:
                assert all(torch.equal(v, initial_weights[j][k]) for j,m in enumerate(init) for k,v in m.items())
            norm = deepcopy(agent.obs_normalizer.state())
            prior = deepcopy(agent.prior.state_dict())
            env = CommandEnv(days, dyn, mode)
            obs = env.reset(eps)
            normed = agent.obs_normalizer(obs)
            assert np.array_equal(normed[:, :11], agent.obs_normalizer.base(obs[:, :11]))
            assert np.array_equal(normed[:, 11:], obs[:, 11:])
            with torch.no_grad():
                initial_action = agent.act_det(obs)
                expected = agent.prior(agent._t(normed[:, :11])).clamp(-1, 1).numpy()
                assert np.array_equal(initial_action, expected)
            buf, diag = collect(agent, c, env, splits['train'], np.random.default_rng(0))
            assert buf['obs'].shape == (6080, 16) and buf['obs_next'].shape == (6080, 16)
            assert np.all(buf['obs'].reshape(760,8,16)[::95,:,11:] == 0)
            if mode == 'padded':
                assert np.all(buf['obs'][:,11:] == 0)
                control = deepcopy(buf)
            else:
                assert np.any(buf['obs'][:,11:] != 0)
                for key in ('act','act_exec','logp','rew','cost','cost_parts','end','terminated'):
                    assert np.array_equal(buf[key],control[key]), key
                assert np.array_equal(buf['obs'][:,:11], control['obs'][:,:11])
            info = agent.update(buf)
            assert all(np.isfinite(v) for v in info.values())
            assert all(torch.isfinite(p).all() for m in modules(agent) for p in m.parameters())
            assert any(not torch.equal(v, init[0][k]) for k,v in agent.pi.state_dict().items())
            if isinstance(agent, PPOFCSRLAgent):
                assert any(not torch.equal(v, init[3][k]) for k,v in agent.encoder.state_dict().items())
            agent.update_constraints([diag['nh4_cost'],diag['do_cost']])
            assert agent.obs_normalizer.state() == norm
            assert all(torch.equal(v,prior[k]) for k,v in agent.prior.state_dict().items())
            checkpoint = a.out / method / mode
            agent.save(checkpoint)
            restored = make_agent(method, 0, a.prepared, c)
            restored.load(checkpoint)
            probe = env.reset(eps)
            assert np.array_equal(agent.act_det(probe), restored.act_det(probe))
            summary, rows, states, air = evaluate(env, splits['train'][:1], lambda x: restored.act_det(x)[:,0])
            assert states.shape == (5,95,5) and air.shape == (5,95)
            assert np.isclose(summary['air_m3'], air.sum(1).mean()*.25)
            assert np.isclose(summary['nh4_cost'], (states[:,:,0]>4).sum(1).mean())
            record = {'method':method,'mode':mode,'seconds':time.monotonic()-start,
                      'parameters':sum(p.numel() for m in modules(agent) for p in m.parameters()),
                      'finite_update':True,'frozen_prior_and_norm':True,'checkpoint_action_exact':True,
                      'train_diagnostic':diag,'update':info}
            records.append(record)
            print('PASS',method,mode,round(record['seconds'],2),flush=True)
    result = {'passed':True,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'records':records,'input_boundary_checks':True,'matched_initial_weights_and_rollout':True,
              'source_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
              'input_sha256':{str(p):sha(p) for p in [a.prepared/'data.pkl',a.prepared/'prior.pt',a.prepared/'normalizer.json',a.corrected]},
              'val_evaluated':False,'test_evaluated':False,'diagnostic_updates_per_arm':1}
    save(a.out/'verification.json',result)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--corrected',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
