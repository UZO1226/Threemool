"""e23 저장 FCSRL의 train trajectory에서 max 목표의 위반 지속시간 구분 여부를 진단한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from fcsrl.utils import DeviceConfig
from recovery import make_agent, recovery_targets
from env import AerationEnv
from evaluate import batched_rollout, episodes
from ppo_config import PPOConfig
from ppo_fcsrl import feasibility_targets

p = argparse.ArgumentParser()
for name in ('data', 'model', 'prepared', 'runs', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
DeviceConfig().select_device(-1)
result = []
for seed in (0, 1):
    agent = make_agent('fcsrl', seed, a.prepared, PPOConfig(total_steps=200000, n_chunks=2, init_log_std=-1.6))
    agent.load(a.runs / 'fcsrl' / f'seed_{seed}' / 'model_final')
    env = AerationEnv(a.data, a.model, split='train'); env.fixed = episodes(env)
    costs = []
    def observe(t, obs, act, cost, air):
        if cost is not None:
            costs.append(cost.copy())
    batched_rollout(env, env.sur, lambda o: agent.act_det(o)[:, 0], 15., observe=observe)
    cost = np.asarray(costs, np.float32); end = np.zeros_like(cost); end[-1] = 1
    maximum = feasibility_targets(cost, np.zeros_like(cost), end, end)
    recovery, valid = recovery_targets(cost, end)
    violating = (cost > 0) & valid.astype(bool)
    collapse = violating & np.isclose(maximum, cost)
    same_current = collapse & (cost == 1)
    x = recovery[same_current]
    row = {'seed': seed, 'split': 'train', 'episodes': len(env.fixed),
           'violating_full_windows': int(violating.sum()),
           'max_equals_current_fraction': float(collapse.sum() / violating.sum()),
           'one_cost_collapsed_windows': len(x), 'recovery_q10_q50_q90': np.quantile(x, [.1, .5, .9]).tolist(),
           'recovery_std': float(x.std()), 'within_hour_recovery_fraction': float((x < 1.).mean())}
    result.append(row); print(json.dumps(row), flush=True)
a.out.write_text(json.dumps(result, indent=2))
