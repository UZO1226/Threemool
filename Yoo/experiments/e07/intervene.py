"""대리 모델의 장시간 반응과 encoder 교체의 정책 변화를 추가 점검한다."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/wwtp'))
import numpy as np
import torch
from agents import load_agent
from env import AerationEnv
from surrogate import do_inputs, nh4_inputs, phase_onehot


def advance(sur, nh4, do, air, phase, temp, q, hour, member):
    ph = phase_onehot(phase)
    s, c = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    dn = np.maximum(0, do + sur.do_model.predict(do_inputs(do, air, ph, temp, nh4, q, s, c), member)[0])
    nn = np.maximum(0, nh4 + sur.nh4_model.predict(nh4_inputs(nh4, do, dn, ph, temp, q, s, c), member)[0])
    return nn, dn


def main(a):
    torch.set_num_threads(1)
    env = AerationEnv(a.data, a.model, split='val')
    report = {'training_performed': False, 'test_used': False,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    days = list(env.day_data.values())
    x = {k: np.stack([d[k] for d in days]) for k in ('nh4', 'do', 'air', 'phase', 'temp', 'q_in', 'hour')}
    actual_air = x['air'][:, :-1]
    env_air = np.where(x['phase'][:, :-1] == 2, actual_air.clip(0, 8000), 0)
    report['replay_action_check'] = {'actual_air_m3': float(actual_air.sum(1).mean()/12),
        'environment_air_m3': float(env_air.sum(1).mean()/12),
        'changed_steps_fraction': float((actual_air != env_air).mean())}
    report['exact_historical_replay'] = []
    measured_cost = ((x['nh4'][:, 1:] > 4).astype(float) + (x['do'][:, 1:] > 2)).sum(1)
    for member in range(5):
        n, d = x['nh4'][:, 0].copy(), x['do'][:, 0].copy()
        costs = np.zeros(len(n))
        errors = []
        for t in range(env.T):
            n, d = advance(env.sur, n, d, x['air'][:, t], x['phase'][:, t], x['temp'][:, t],
                           x['q_in'][:, t], x['hour'][:, t], member)
            costs += (n > 4).astype(float) + (d > 2)
            errors.append(np.abs(n - x['nh4'][:, t+1]))
        report['exact_historical_replay'].append({'member': member, 'predicted_cost': float(costs.mean()),
            'measured_cost': float(measured_cost.mean()), 'nh4_mae': float(np.mean(errors))})
    # 검증의 실제 상태에서 시작해 동일한 외생 조건으로 3시간 동안 비교한다.
    starts = [(i, t) for i, day in enumerate(days) for t in range(0, env.T-35, 6)
              if day['phase'][t] == 2 and day['air'][t] <= 7000]
    di, ti = np.array(starts).T
    high = x['nh4'][di, ti] > 4
    report['air_intervention'] = []
    for member in range(5):
        trajectories = []
        for extra in (0, 1000):
            n, d = x['nh4'][di, ti].copy(), x['do'][di, ti].copy()
            snapshots = {}
            for step in range(36):
                ix = ti + step
                ph = x['phase'][di, ix]
                air = np.where(ph == 2, np.clip(x['air'][di, ix]+extra, 0, 8000), 0)
                n, d = advance(env.sur, n, d, air, ph, x['temp'][di, ix], x['q_in'][di, ix], x['hour'][di, ix], member)
                if step + 1 in (1, 12, 36): snapshots[step+1] = (n.copy(), d.copy())
            trajectories.append(snapshots)
        for horizon in (1, 12, 36):
            delta = trajectories[1][horizon][0] - trajectories[0][horizon][0]
            for group, mask in (('all', np.ones(len(high), dtype=bool)), ('initial_nh4_gt4', high)):
                report['air_intervention'].append({'member': member, 'minutes': horizon*5, 'group': group,
                    'samples': int(mask.sum()), 'nh4_change_mean': float(delta[mask].mean()),
                    'nh4_increases_fraction': float((delta[mask] > 1e-6).mean())})
    report['encoder_swap_policy_change'] = []
    for seed in (0, 1):
        run = a.runs / 'ppolag_fcsrl' / f'seed_{seed}'
        agent, earlier = load_agent(run, env, cudaid=-1, checkpoint='final'), load_agent(run, env, cudaid=-1, checkpoint='best')
        state = {k:v.clone() for k,v in agent.fixed_encoder.state_dict().items()}
        obs = np.load(a.diagnosis / f'ppolag_fcsrl_s{seed}_val.npz')['obs'][:-1].reshape(-1,11)
        mu = []
        with torch.no_grad():
            for encoder_state in (earlier.fixed_encoder.state_dict(), state):
                agent.fixed_encoder.load_state_dict(encoder_state)
                out = []
                for o in np.array_split(obs, 32):
                    out.append(agent.dist(agent._t(agent.obs_normalizer(o))).mean.cpu().numpy())
                mu.append(np.concatenate(out).reshape(-1))
        std = float(agent.log_std.exp().item())
        kl = (mu[1] - mu[0])**2 / (2*std**2)
        air = 4000*(np.clip(mu[1],-1,1) - np.clip(mu[0],-1,1))
        mask = obs[:,4] == 1
        report['encoder_swap_policy_change'].append({'seed': seed, 'kl_mean': float(kl.mean()),
            'kl_p95': float(np.quantile(kl,.95)), 'kl_max': float(kl.max()),
            'aerobic_air_final_minus_earlier_mean': float(air[mask].mean()),
            'aerobic_air_absolute_difference_mean': float(np.abs(air[mask]).mean())})
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
