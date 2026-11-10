"""e30 환경에서 두 제약을 각각 갱신한다. 관측, 시간 계수와 표현 목표는 고정한다."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm
from dual import config, make_agent, METHODS, LIMITS, save
from fcsrl.utils import DeviceConfig
from ppo_lag import gae
from ppo_fcsrl import PPOFCSRLAgent, feasibility_targets, RepresentationConfig
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes


def collect(agent, c, env, days, rng):
    T, N = c.horizon, c.n_env
    assert T % env.T == 0
    member = int(rng.integers(env.dyn.n_members))
    b = {k:np.zeros((T,N),np.float32) for k in
         ('logp','rew','cost','vr','nvr','end','terminated','next_feasi')}
    for k in ('obs','obs_next'):
        b[k] = np.zeros((T,N,11),np.float32)
    for k in ('act','act_exec'):
        b[k] = np.zeros((T,N,1),np.float32)
    for k in ('cost_parts','vc','nvc'):
        b[k] = np.zeros((T,N,2),np.float32)
    def reset():
        return env.reset([(days[i],member) for i in rng.integers(len(days),size=N)])
    obs = reset()
    for t in range(T):
        on = agent.obs_normalizer(obs)
        with torch.no_grad():
            ot = agent._t(on)
            dist = agent.dist(ot)
            action = dist.sample()
            b['logp'][t] = dist.log_prob(action).sum(-1).numpy()
            vr, vc = agent.values(ot)
            b['vr'][t], b['vc'][t] = vr.numpy(), vc.numpy()
        b['obs'][t], b['act'][t] = on, action.numpy()
        b['act_exec'][t] = np.clip(b['act'][t],-1,1)
        nxt, reward, cost, trunc, info = env.step(b['act_exec'][t,:,0])
        b['rew'][t], b['cost'][t], b['end'][t] = reward, cost, float(trunc)
        b['cost_parts'][t] = np.column_stack([info['v_nh4'],info['v_do']])
        no = agent.obs_normalizer(nxt)
        b['obs_next'][t] = no
        with torch.no_grad():
            vr, vc = agent.values(agent._t(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(),vc.numpy()
            if isinstance(agent,PPOFCSRLAgent):
                b['next_feasi'][t] = agent.next_feasibility(no)
        obs = reset() if trunc and t+1<T else nxt
    assert np.array_equal(b['cost'],b['cost_parts'].sum(-1))
    b['adv_r'],b['ret_r'] = gae(b['rew'],b['vr'],b['nvr'],b['end'],c.gamma,c.gae_lambda)
    parts = [gae(b['cost_parts'][:,:,j],b['vc'][:,:,j],b['nvc'][:,:,j],b['end'],c.gamma,c.gae_lambda) for j in range(2)]
    b['adv_c'],b['ret_c'] = np.stack([v[0] for v in parts],-1),np.stack([v[1] for v in parts],-1)
    b['feasi'] = feasibility_targets(b['cost'],b['next_feasi'],b['end'],b['terminated'],RepresentationConfig().feasibility_discount)
    costs = b['cost_parts'].reshape(-1,env.T,N,2).sum(1).mean((0,1))
    diag = {'member':member,'nh4_cost':float(costs[0]),'do_cost':float(costs[1]),
            'raw_action_clip_percent':float((abs(b['act'])>1).mean()*100)}
    return {k:v.reshape(T*N,*v.shape[2:]) for k,v in b.items()},diag


def evaluate(env, days, act):
    obs = env.reset(all_episodes(days,env.dyn.n_members))
    n = len(obs)
    air, costs = np.zeros(n),np.zeros((n,2))
    for _ in range(env.T):
        obs,_,_,_,info = env.step(act(obs))
        air += info['air']*.25
        costs += np.column_stack([info['v_nh4'],info['v_do']])
    rows = [{'day':d,'member':int(m),'air_m3':float(air[i]),'nh4_cost':float(costs[i,0]),'do_cost':float(costs[i,1])}
            for i,(d,m) in enumerate(env.episodes)]
    summary = {k:float(np.mean([x[k] for x in rows])) for k in ('air_m3','nh4_cost','do_cost')}
    summary.update(cost=float(costs.sum(1).mean()),both_mean_limits=bool(np.all(costs.mean(0)<=LIMITS)),
        both_limits_episode_pct=float(100*np.all(costs<=LIMITS,axis=1).mean()),n_episodes=n)
    return summary,rows


def train(a):
    a.out.mkdir(parents=True,exist_ok=False)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    c = config(a.updates)
    agent = make_agent(a.method,a.seed,a.prepared,c)
    splits,days = load_days(a.prepared/'data.pkl')
    env = BatchEnv(days,Dynamics(a.prepared/'surrogate.pkl'))
    save(a.out/'effective_config.json',{'method':a.method,'seed':a.seed,'config':c.to_dict(),
        'cost_limits':LIMITS.tolist(),'feasibility':'e30 summed cost unchanged','prepared':str(a.prepared),
        'selection':'last','split':'val','torch_threads':torch.get_num_threads()})
    norm = agent.obs_normalizer.state()
    prior = {k:v.clone() for k,v in agent.prior.state_dict().items()}
    start,records = time.monotonic(),[]
    rng = np.random.default_rng(a.seed)
    for u in tqdm(range(1,c.n_updates+1),desc=f'dual {a.method} s{a.seed}',mininterval=10):
        buf,diag = collect(agent,c,env,splits['train'],rng)
        info = agent.update(buf)
        assert all(np.isfinite(v) for v in info.values())
        modules = [agent.pi,agent.vr,agent.vc]
        if isinstance(agent,PPOFCSRLAgent):
            modules += [agent.encoder,agent.fixed_encoder,agent.feasi_head,agent.proj_layer,agent.post_proj]
        assert all(torch.isfinite(p).all() for m in modules for p in m.parameters())
        agent.update_constraints([diag['nh4_cost'],diag['do_cost']])
        records.append({'update':u,'lambda_nh4':float(agent.multipliers[0]),'lambda_do':float(agent.multipliers[1]),**diag,**info})
        elapsed = time.monotonic()-start
        save(a.out/'train_log.json',records)
        save(a.out/'progress.json',{'phase':'training','update':u,'updates':c.n_updates,'steps':u*6080,
             'total_steps':c.actual_steps,'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/u*(c.n_updates-u)})
    assert norm==agent.obs_normalizer.state()
    assert all(torch.equal(v,prior[k]) for k,v in agent.prior.state_dict().items())
    agent.save(a.out/'model_final')
    summary,rows = evaluate(env,splits['val'],lambda x:agent.act_det(x)[:,0])
    save(a.out/'val_episodes.json',rows)
    save(a.out/'metrics.json',{'method':a.method,'seed':a.seed,'val_final':summary,'steps':c.actual_steps,
         'normalizer_and_prior_frozen':True,'elapsed_seconds':time.monotonic()-start})
    save(a.out/'progress.json',{'phase':'complete','steps':c.actual_steps})
    print('DONE',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--updates',type=int,default=66)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    train(p.parse_args())
