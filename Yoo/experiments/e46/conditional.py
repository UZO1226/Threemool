"""e44 FCSRL에 에피소드 시작 조건별 NH4 승수만 적용한다."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'e44'))
from command import CommandEnv, CommandFCSRL, make_agent as original_agent, config, save, sha
from train import collect as original_collect, evaluate
from dual import combine_advantage
from fcsrl.utils import PIDLagrangianUpdater, DeviceConfig
sys.path.insert(0, str(HERE.parent / 'e45'))
from evaluate import load_selected, metrics as detailed_metrics
from darrow_env import Dynamics, all_episodes
sys.path.insert(0, str(HERE))


def group_definition(splits, days):
    initial = np.array([days[d]['exo'][0, 1] for d in splits['train']])
    threshold = float(np.quantile(initial, .25))
    group = {d: int(days[d]['exo'][0, 1] > threshold) for d in splits['train'] + splits['val']}
    counts = np.bincount([group[d] for d in splits['train']], minlength=2)
    assert np.all(counts > 0)
    return dict(threshold=threshold, definition='initial temperature <= train q25 is cold (0); otherwise normal (1)',
                counts=counts.tolist(), probabilities=(counts / counts.sum()).tolist(), group=group)


class TrackedEnv(CommandEnv):
    def __init__(self, days, dynamics, definition):
        super().__init__(days, dynamics, 'padded')
        self.definition = definition
        self.reset_groups = []

    def reset(self, episodes):
        episodes = list(episodes)
        self.reset_groups.append(np.array([self.definition['group'][d] for d, _ in episodes]))
        return super().reset(episodes)


def collect(agent, c, env, days, rng):
    env.reset_groups = []
    buf, diag = original_collect(agent, c, env, days, rng)
    groups = np.stack(env.reset_groups)
    assert groups.shape == (c.horizon // env.T, c.n_env)
    buf['group'] = np.repeat(groups, env.T, axis=0).reshape(-1)
    costs = buf['cost_parts'].reshape(-1, env.T, c.n_env, 2).sum(1)
    diag['group_counts'] = np.bincount(groups.ravel(), minlength=2).tolist()
    diag['group_nh4_costs'] = [float(costs[:, :, 0][groups == g].mean()) if np.any(groups == g) else None for g in range(2)]
    assert np.isclose(costs[:, :, 0].mean(), diag['nh4_cost'])
    return buf, diag


def conditional_advantage(ar, ac, groups, lambdas, probabilities):
    ar = (ar - ar.mean()) / (ar.std() + 1e-8)
    ac = ac - ac.mean(0, keepdims=True)
    scale = 1 + np.dot(probabilities, lambdas[:2]) + lambdas[2]
    return (ar - lambdas[np.asarray(groups)] * ac[:, 0] - lambdas[2] * ac[:, 1]) / scale


class ConditionalFCSRL(CommandFCSRL):
    def update_group_constraints(self, diag):
        costs = diag['group_nh4_costs'] + [diag['do_cost']]
        for pid, cost in zip(self.group_pids, costs):
            if cost is not None:
                pid.update(cost, 6.)
        self.group_lambdas = np.array([p.get_lagrg() for p in self.group_pids])

    def save(self, path):
        super().save(path)
        save(Path(path) / 'constraints.json', dict(names=['NH4_cold', 'NH4_normal', 'DO_global'],
             limits=[6., 6., 6.], multipliers=self.group_lambdas.tolist(),
             group_probabilities=self.group_probabilities.tolist(), feasibility_target='unchanged summed binary costs'))

    def load(self, path):
        super().load(path)
        state = json.loads((Path(path) / 'constraints.json').read_text())
        self.group_lambdas = np.asarray(state['multipliers'])
        self.group_probabilities = np.asarray(state['group_probabilities'])

    def update(self, buf):
        # e33 DualMixin.update 그대로, 결합 advantage만 조건별 승수로 바꾼다.
        c = self.config
        o, action, old_logp = [self._t(buf[k]) for k in ('obs', 'act', 'logp')]
        adv = self._t(conditional_advantage(buf['adv_r'], buf['adv_c'], buf['group'],
                                           self.group_lambdas, self.group_probabilities))
        rr, rc = self._t(buf['ret_r']), self._t(buf['ret_c'])
        self._sync_probe = o[:1024]
        aux = {}
        try:
            for ep in range(c.update_epochs):
                perm = torch.randperm(len(o), device=self.dev)
                kls = []
                for start in range(0, len(o), c.minibatch_size):
                    idx = perm[start:start + c.minibatch_size]
                    for key, value in self.update_aux(buf, idx).items():
                        aux.setdefault(key, []).append(value)
                    dist = self.dist(o[idx])
                    logp = dist.log_prob(action[idx]).sum(-1)
                    ratio = (logp - old_logp[idx]).clamp(-20, 20).exp()
                    clipped = ratio.clamp(1 - c.clip_ratio, 1 + c.clip_ratio)
                    loss_pi = -torch.minimum(ratio * adv[idx], clipped * adv[idx]).mean()
                    self.opt_pi.zero_grad()
                    loss_pi.backward()
                    nn.utils.clip_grad_norm_(list(self.pi.parameters()) + [self.log_std], c.max_grad_norm)
                    self.opt_pi.step()
                    vr, vc = self.values(o[idx])
                    loss_v = ((vr - rr[idx]) / c.value_scale).square().mean()
                    loss_v = loss_v + ((vc - rc[idx]) / c.value_scale).square().sum(-1).mean()
                    self.opt_v.zero_grad()
                    loss_v.backward()
                    nn.utils.clip_grad_norm_(list(self.vr.parameters()) + list(self.vc.parameters()), c.max_grad_norm)
                    self.opt_v.step()
                    kls.append(float((old_logp[idx] - logp).mean().detach()))
                if np.mean(kls) > c.target_kl:
                    break
            result = self.finish_update() or {}
            result.update({key: float(np.mean(v)) for key, v in aux.items()})
            result.update(kl=float(np.mean(kls)), pi_epochs=ep + 1,
                          std=float(self.log_std.detach().exp().mean()))
            return result
        finally:
            del self._sync_probe


def make_agent(arm, seed, prepared, c, definition):
    a = original_agent('fcsrl', seed, prepared, c)
    if arm == 'conditional':
        a.__class__ = ConditionalFCSRL
        a.group_probabilities = np.asarray(definition['probabilities'])
        a.group_lambdas = np.zeros(3)
        a.group_pids = [PIDLagrangianUpdater(0., [c.pid_kp, c.pid_ki, c.pid_kd], c.max_lambda) for _ in range(3)]
    else:
        assert arm == 'baseline'
    return a


def modules(a):
    return [getattr(a, k) for k in ('pi', 'vr', 'vc', 'encoder', 'fixed_encoder', 'feasi_head', 'proj_layer', 'post_proj', 'prior')]


def input_hashes(prepared, corrected):
    return {str(p): sha(p) for p in [prepared / 'data.pkl', prepared / 'prior.pt', prepared / 'normalizer.json', corrected]}


def source_hashes():
    files = [*HERE.glob('*.py'), * (ROOT / 'Yoo/code/wwtp').glob('*.py')]
    for folder in ('e23', 'e25', 'e29', 'e30', 'e33', 'e44', 'e45'):
        files += list((HERE.parent / folder).glob('*.py'))
    return {str(p): sha(p) for p in files}
