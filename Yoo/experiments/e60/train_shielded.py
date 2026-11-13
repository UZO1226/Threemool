"""e60: Park/e12 방식으로 최소 개입 차단을 학습 중에 켠 Yoo PPO-Lag / FCSRL 학습.

e56 hybrid 실행기(CPU1 수집, PPO 는 CPU1 갱신, FCSRL 은 GPU1 갱신)를 그대로 쓰고, 수집에서 실행 행동만
Park 차단(e59 에 복사한 min_brake.py, 1스텝 다른 구성원 target 3.5)을 거친다. Park/e09, e12 와 같이
PPO 확률비는 정책이 뽑은 행동으로, 보상, 비용, FCSRL 표현 학습은 실제로 넣은 행동으로 계산한다.
학습 뒤 검증 40일 x 구성원 5개를 차단 없음, park_e10, park_e19 로 각각 평가한다. test 행은 쓰지 않는다.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e59'))
import shield_eval as se  # noqa: E402  e59: base(e56 train), s(e58 severity), min_brake, SHIELDS, rollout
base, h = se.base, se.s.h
from ppo_lag import gae  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent, feasibility_targets, RepresentationConfig  # noqa: E402

TRAIN_SHIELD = 'park_e10'
EVAL_SHIELDS = ('none', 'park_e10', 'park_e19')


def collect(agent, c, env, days, rng, shield):
    """e30 collect 와 같고, 실행 행동만 차단을 거친다. shield=None 이면 e30 과 정확히 같은 buffer 를 만든다."""
    T, N = c.horizon, c.n_env
    assert T % env.T == 0
    member = int(rng.integers(env.dyn.n_members))
    b = {k: np.zeros((T, N), np.float32) for k in
         ('logp', 'rew', 'cost', 'vr', 'vc', 'nvr', 'nvc', 'end', 'terminated', 'next_feasi')}
    for k in ('obs', 'obs_next'):
        b[k] = np.zeros((T, N, 11), np.float32)
    for k in ('act', 'act_exec'):
        b[k] = np.zeros((T, N, 1), np.float32)

    def reset():
        return env.reset([(days[i], member) for i in rng.integers(len(days), size=N)])
    obs = reset()
    used_steps = 0
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
        a_exec = np.clip(b['act'][t, :, 0], -1, 1)
        if shield is not None:
            a_exec, used = shield(obs, a_exec, env)
            used_steps += int(used.sum())
        b['act_exec'][t, :, 0] = a_exec
        nxt, reward, cost, trunc, info = env.step(b['act_exec'][t, :, 0])
        b['rew'][t], b['cost'][t], b['end'][t] = reward, cost, float(trunc)
        no = agent.obs_normalizer(nxt)
        b['obs_next'][t] = no
        with torch.no_grad():
            vr, vc = agent.values(agent._t(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(), vc.numpy()
            if isinstance(agent, PPOFCSRLAgent):
                b['next_feasi'][t] = agent.next_feasibility(no)
        obs = reset() if trunc and t + 1 < T else nxt
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    b['feasi'] = feasibility_targets(b['cost'], b['next_feasi'], b['end'], b['terminated'],
                                     RepresentationConfig().feasibility_discount)
    diag = {'member': member, 'daily_cost_mean': float(b['cost'].reshape(-1, env.T, N).sum(1).mean()),
            'raw_action_clip_percent': float((abs(b['act']) > 1).mean() * 100),
            'reward_mean': float(b['rew'].mean()), 'intervention_percent': 100 * used_steps / (T * N),
            'policy_air_mean': float(se.action_to_air(np.clip(b['act'][:, :, 0], -1, 1)).mean()),
            'executed_air_mean': float(se.action_to_air(b['act_exec'][:, :, 0]).mean())}
    return {k: v.reshape(T * N, *v.shape[2:]) for k, v in b.items()}, diag


def run(a):
    manifest = json.loads(a.manifest.read_text())
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items())
    if a.resume:
        assert (a.out / 'resume.pt').exists()
    else:
        a.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0' if a.method == 'fcsrl' else 'cpu'
    c = base.common.config(a.updates)
    learner, collector = h.make_pair(a.method, a.seed, a.prepared, c, 'on', device)
    rng = np.random.default_rng(a.seed)
    metadata = dict(method=a.method, seed=a.seed, prior_mode='on', backend=device, updates=a.updates, train_shield=a.train_shield,
                    code_commit=manifest['code_commit'], data_sha256=base.sha(a.prepared / 'data.pkl'),
                    model_sha256=base.sha(a.prepared / 'surrogate.pkl'), min_brake_sha256=base.sha(se.HERE / 'min_brake.py'))
    records = h.restore(a.out / 'resume.pt', learner, rng, metadata) if a.resume else []
    base.save(a.out / 'effective_config.json', dict(**metadata, config=c.to_dict(), cpu_threads=1,
              gpu_physical_index=1 if device != 'cpu' else None, train_shield_config=se.jsonable(se.SHIELDS[a.train_shield]),
              eval_shields=list(EVAL_SHIELDS), cost_limit=12, observation_dim=11, test_evaluated=False, prepared=str(a.prepared)))
    splits, days = base.load_selected(a.prepared / 'data.pkl')
    dyn = base.Dynamics(a.prepared / 'surrogate.pkl')
    env = base.BatchEnv(days, dyn)
    train_shield = se.make_shield(a.train_shield, dyn)
    original_prior = {k: v.detach().cpu().clone() for k, v in learner.prior.state_dict().items()}
    norm = learner.obs_normalizer.state()
    start = time.monotonic()
    initial_updates = len(records)
    for u in tqdm(range(initial_updates + 1, c.n_updates + 1), desc=f'e60 {a.method} s{a.seed}', mininterval=10):
        tick = time.monotonic()
        h.sync_collector(learner, collector)
        h.configure('cpu')
        buf, diag = collect(collector, c, env, splits['train'], rng, train_shield)
        collection = time.monotonic() - tick
        tick = time.monotonic()
        info = h.update(learner, buf)
        updating = time.monotonic() - tick
        learner.pid.update(diag['daily_cost_mean'], c.cost_limit)
        learner.lagrg = learner.pid.get_lagrg()
        records.append(dict(update=u, **{'lambda': learner.lagrg}, **diag, **info, collection_seconds=collection, update_seconds=updating))
        h.checkpoint(a.out / 'resume.pt', learner, rng, records, metadata)
        base.save(a.out / 'train_log.json', records)
        elapsed = time.monotonic() - start
        base.save(a.out / 'progress.json', dict(phase='training', update=u, updates=c.n_updates, steps=u * 6080, total_steps=c.actual_steps,
                  elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / (u - initial_updates) * (c.n_updates - u), backend=device))
    assert learner.obs_normalizer.state() == norm
    assert all(torch.equal(v.detach().cpu(), original_prior[k]) for k, v in learner.prior.state_dict().items())
    h.sync_collector(learner, collector)
    h.configure('cpu')
    collector.save(a.out / 'model_final')
    restored = base.common.make_agent(a.method, a.seed, a.prepared, c)
    restored.load(a.out / 'model_final')
    assert base.fingerprint(restored) == base.fingerprint(collector)
    fp = base.fingerprint(collector)
    act = lambda obs: collector.act_det(obs)[:, 0]  # noqa: E731
    summaries = {}
    for split, names in (('val', EVAL_SHIELDS), ('train', ('none', TRAIN_SHIELD))):
        for name in names:
            base.save(a.out / 'progress.json', dict(phase=f'evaluating_{split}_{name}', steps=c.actual_steps))
            result, rows, members, states, airs = se.rollout(env, splits[split], act, se.make_shield(name, dyn))
            assert base.fingerprint(collector) == fp
            summaries[f'{split}_{name}'] = result
            base.save(a.out / f'{split}_{name}_episodes.json', rows)
            np.savez_compressed(a.out / f'{split}_{name}_rollout.npz', states=states, air=airs)
    base.save(a.out / 'metrics.json', dict(method=a.method, seed=a.seed, prior_mode='on', train_shield=a.train_shield, **summaries,
              val_final=summaries['val_' + TRAIN_SHIELD], steps=c.actual_steps, normalizer_and_prior_frozen=True, test_evaluated=False,
              backend=device, policy_fingerprint=fp, final_lambda=learner.lagrg, elapsed_seconds=time.monotonic() - start))
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items())
    base.save(a.out / 'progress.json', dict(phase='complete', steps=c.actual_steps, elapsed_seconds=time.monotonic() - start))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ppo', 'fcsrl'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--train-shield', default=TRAIN_SHIELD, choices=[k for k in se.SHIELDS if k != 'none'])
    p.add_argument('--prepared', type=Path, default=se.PREPARED)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())
