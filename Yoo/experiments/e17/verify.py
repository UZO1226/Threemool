"""비용 단위, 종료 경계와 실제 BSM 전이의 최적화 연결을 검사한다."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
import torch
from bsm_mean import normalized_cost, MeanEnv, episode_metrics
from train_pilot import make_agent, collect, checked_update
from ppo_lag import gae
from ppo_fcsrl import feasibility_targets
from fcsrl.utils import DeviceConfig, PIDLagrangianUpdater

p = argparse.ArgumentParser()
p.add_argument('--prepared', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
warnings.filterwarnings('error', category=ODEintWarning)
torch.set_num_threads(1)
DeviceConfig().select_device(-1)
q = np.array([[1., 3.], [3., 1.]])
n = np.array([[2., 2.], [6., 6.]])
c = normalized_cost(n * q / 4, q)
np.testing.assert_allclose(c.mean(axis=0), [1.25, .75])
# 유량 단위 전체를 바꿔도 같은 농도 제약이어야 한다.
np.testing.assert_allclose(normalized_cost(n * q * 100 / 4, q * 100), c)
old = PIDLagrangianUpdater(0., [.02, .005, .01], 20.)
new = PIDLagrangianUpdater(0., np.array([.02, .005, .01]) * 15 / 288, 20.)
for ratio in [1.2, 1.1, .6, 1.3, 1.5]:
    old.update(ratio * 15, 15)
    new.update(ratio * 288, 288)
    np.testing.assert_allclose(new.get_lagrg(), old.get_lagrg(), atol=1e-14)
reward = np.array([[1.], [100.]])
end = np.ones_like(reward)
adv, _ = gae(reward, np.zeros_like(reward), np.zeros_like(reward), end, 1., 1.)
np.testing.assert_array_equal(adv, reward)
f = feasibility_targets(reward, np.ones_like(reward) * 1e6, end, end)
np.testing.assert_array_equal(f, reward)
state = dict(np.load(a.prepared / 'day_0.npz'))
streams = []
for action in [-1., 1.]:
    env = MeanEnv(state)
    env.reset()
    infos = []
    for _ in range(env.T):
        _, _, _, _, info = env.step(np.array([action]))
        infos.append(info)
    cost = normalized_cost(np.array([i['load'] for i in infos]), np.array([i['flow_ratio'] for i in infos]))
    metrics = episode_metrics(infos)
    np.testing.assert_allclose(cost.mean(), metrics['cost_ratio'], rtol=1e-12)
    streams.append(np.concatenate([i['effluent'][:, 14] for i in infos]))
np.testing.assert_array_equal(*streams)
agent, config, tail = make_agent('fcsrl_tail', 999, np.load(a.prepared / 'train_observations.npy'), updates=1)
buf, diag = collect(agent, config, tail, a.prepared, np.random.default_rng(999))
assert diag['feasibility_symlog_outside_fraction'] == 0
update = checked_update(agent, buf)
assert update['policy_parameter_delta_l2'] > 0
summary = {'cost_ratio_identity': True, 'flow_unit_invariance': True, 'pid_relative_error_invariance': True,
    'terminal_gae_and_feasibility_no_bootstrap': True, 'flow_independent_of_extreme_actions': True,
    'real_bsm_fcsrl_tail_optimizer_update': True, 'preflight_seed': 999,
    'not_part_of_comparison': True, 'diagnostics': diag, 'update': update}
a.out.write_text(json.dumps(summary, indent=2, allow_nan=False))
print(json.dumps(summary, indent=2), flush=True)
