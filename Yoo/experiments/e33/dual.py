"""NH4/DO 비용 가치와 PID를 분리한다. FCSRL 표현 학습은 e30 그대로 유지한다."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from common import config, make_agent as base_agent, save
from behavior import PriorPPO, PriorFCSRL
from ablation import AblatedFCSRL
from ppo_lag import mlp
from ppo_fcsrl import PPOFCSRLAgent
from fcsrl.utils import PIDLagrangianUpdater
sys.path.insert(0, str(HERE))

METHODS = ('ppo', 'fcsrl', 'fcsrl_actor')
LIMITS = np.array([6., 6.])


def combine_advantage(ar, ac, multipliers):
    ar = (ar-ar.mean())/(ar.std()+1e-8)
    ac = ac-ac.mean(0, keepdims=True)
    return (ar-ac @ np.asarray(multipliers))/(1+np.sum(multipliers))


class DualMixin:
    def values(self, o):
        h = self.policy_input(o)
        return self.vr(h)[:, 0]*self.config.value_scale, self.vc(h)*self.config.value_scale

    def update_constraints(self, costs):
        for pid, cost, limit in zip(self.constraint_pids, costs, LIMITS):
            pid.update(cost, limit)
        self.multipliers = np.array([p.get_lagrg() for p in self.constraint_pids])

    def update(self, buf):
        c = self.config
        o, action, old_logp = [self._t(buf[k]) for k in ('obs','act','logp')]
        adv = self._t(combine_advantage(buf['adv_r'], buf['adv_c'], self.multipliers))
        rr, rc = self._t(buf['ret_r']), self._t(buf['ret_c'])
        self._sync_probe = o[:1024]
        if isinstance(self, DualActor):
            ids = np.random.default_rng(25000).permutation(len(o))
            nf = min(2048, len(ids)*2//3)
            nc = min(1024, len(ids)-nf)
            self._preserve_obs, self._fit_size = o[ids[:nf+nc]], nf
        aux = {}
        try:
            for ep in range(c.update_epochs):
                perm = torch.randperm(len(o), device=self.dev)
                kls = []
                for start in range(0, len(o), c.minibatch_size):
                    idx = perm[start:start+c.minibatch_size]
                    # Original dynamics/feasibility update and summed-cost target are unchanged.
                    for key, value in self.update_aux(buf, idx).items():
                        aux.setdefault(key, []).append(value)
                    dist = self.dist(o[idx])
                    logp = dist.log_prob(action[idx]).sum(-1)
                    ratio = (logp-old_logp[idx]).clamp(-20, 20).exp()
                    clipped = ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)
                    loss_pi = -torch.minimum(ratio*adv[idx], clipped*adv[idx]).mean()
                    self.opt_pi.zero_grad()
                    loss_pi.backward()
                    nn.utils.clip_grad_norm_(list(self.pi.parameters())+[self.log_std], c.max_grad_norm)
                    self.opt_pi.step()
                    vr, vc = self.values(o[idx])
                    loss_v = ((vr-rr[idx])/c.value_scale).square().mean()
                    loss_v = loss_v + ((vc-rc[idx])/c.value_scale).square().sum(-1).mean()
                    self.opt_v.zero_grad()
                    loss_v.backward()
                    nn.utils.clip_grad_norm_(list(self.vr.parameters())+list(self.vc.parameters()), c.max_grad_norm)
                    self.opt_v.step()
                    kls.append(float((old_logp[idx]-logp).mean().detach()))
                if np.mean(kls) > c.target_kl:
                    break
            result = self.finish_update() or {}
            result.update({key: float(np.mean(v)) for key,v in aux.items()})
            result.update(kl=float(np.mean(kls)), pi_epochs=ep+1,
                          std=float(self.log_std.detach().exp().mean()))
            return result
        finally:
            del self._sync_probe
            if hasattr(self, '_preserve_obs'):
                del self._preserve_obs, self._fit_size

    def save(self, path):
        super().save(path)
        save(Path(path)/'constraints.json', {'names':['NH4','DO'], 'limits':LIMITS.tolist(),
             'multipliers':self.multipliers.tolist(), 'feasibility_target':'unchanged summed binary costs'})


class DualPPO(DualMixin, PriorPPO):
    pass


class DualFCSRL(DualMixin, PriorFCSRL):
    pass


class DualActor(DualMixin, AblatedFCSRL):
    pass


def make_agent(method, seed, prepared, c):
    assert method in METHODS
    a = base_agent(method, seed, prepared, c)
    a.__class__ = {'ppo':DualPPO,'fcsrl':DualFCSRL,'fcsrl_actor':DualActor}[method]
    rng = torch.get_rng_state()
    old = a.vc
    a.vc = mlp(old[0].in_features, 2, c.hidden_dim).to(a.dev)
    with torch.no_grad():
        for i in (0,2):
            a.vc[i].load_state_dict(old[i].state_dict())
        a.vc[-1].weight.copy_(old[-1].weight.repeat(2,1)/2)
        a.vc[-1].bias.copy_(old[-1].bias.repeat(2)/2)
    torch.set_rng_state(rng)
    a.opt_v = torch.optim.Adam(list(a.vr.parameters())+list(a.vc.parameters()), lr=c.critic_lr)
    a.multipliers = np.zeros(2)
    a.constraint_pids = [PIDLagrangianUpdater(0., [c.pid_kp,c.pid_ki,c.pid_kd], c.max_lambda) for _ in range(2)]
    return a
