"""e25의 보정 대상을 분리하고 같은 추가 갱신 예산의 PPO 대조를 제공한다."""
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e25'))
from preserve import (PreservingFCSRL, make_agent as e25_agent,
                      STEPS, FIT_SIZE, CHECK_SIZE, MIN_REDUCTION)

METHODS = ('fcsrl', 'fcsrl_actor', 'fcsrl_critic', 'fcsrl_preserve', 'fcsrl_extra_ppo')


class AblatedFCSRL(PreservingFCSRL):
    def update(self, buf):
        self._extra_buffer = buf
        try:
            return super().update(buf)
        finally:
            del self._extra_buffer

    def compensate(self, features, teachers, scales, nf):
        wall, cpu = time.perf_counter(), time.process_time()
        if self.mode == 'fcsrl_preserve':
            result = super().compensate(features, teachers, scales, nf)
        else:
            result = self._compensate(features, teachers, scales, nf)
        result.update(extra_wall_seconds=time.perf_counter() - wall,
                      extra_cpu_seconds=time.process_time() - cpu)
        return result

    def _compensate(self, features, teachers, scales, nf):
        modules = (self.pi, self.vr, self.vc)
        initial = [deepcopy(m.state_dict()) for m in modules]
        active = {'fcsrl_actor': (0,), 'fcsrl_critic': (1, 2),
                  'fcsrl_extra_ppo': (0, 1, 2)}[self.mode]
        extra_ppo = self.mode == 'fcsrl_extra_ppo'

        def errors(sl):
            terms = [(((m(features[sl]) - t[sl]) / scale) ** 2).mean()
                     for m, t, scale in zip(modules, teachers, scales)]
            terms[0] = .5 * terms[0]
            return torch.stack(terms)

        with torch.no_grad():
            before = errors(slice(nf, None))
            anchor_mean = self.pi(features).detach()
        if extra_ppo:
            buf = self._extra_buffer
            ids = np.random.default_rng(25000).permutation(len(buf['obs']))[:len(features)]
            assert np.array_equal(buf['obs'][ids], self._preserve_obs.cpu().numpy())
            ar, ac = buf['adv_r'], buf['adv_c']
            ar = (ar - ar.mean()) / (ar.std() + 1e-8)
            ac = ac - ac.mean()
            adv = self._t(((ar - self.lagrg * ac) / (1 + self.lagrg))[ids])
            action, old_logp = self._t(buf['act'][ids]), self._t(buf['logp'][ids])
            returns = [self._t(buf[k][ids]) for k in ('ret_r', 'ret_c')]
            prior = self.prior_action(self._preserve_obs).detach()

            def task_losses(sl):
                dist = torch.distributions.Normal(prior[sl] + self.pi(features[sl]), scales[0])
                logp = dist.log_prob(action[sl]).sum(-1)
                ratio = (logp - old_logp[sl]).clamp(-20, 20).exp()
                clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                policy = -torch.minimum(ratio * adv[sl], clipped * adv[sl]).mean()
                values = [((net(features[sl])[:, 0] - ret[sl] / self.config.value_scale) ** 2).mean()
                          for net, ret in zip(modules[1:], returns)]
                return torch.stack([policy, *values])

        score = float(before[list(active)].sum())
        best_score, best_step, best_state, attempts = score, 0, initial, 0
        groups = []
        if 0 in active:
            groups.append({'params': list(self.pi.parameters()), 'lr': self.config.actor_lr})
        if 1 in active:
            groups.append({'params': list(self.vr.parameters()) + list(self.vc.parameters()),
                           'lr': self.config.critic_lr})
        params = [p for group in groups for p in group['params']]
        if extra_ppo or score > 1e-12:
            optimizer = torch.optim.Adam(groups)
            for step in range(1, STEPS + 1):
                loss = task_losses(slice(0, nf)).sum() if extra_ppo else errors(slice(0, nf))[list(active)].sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite extra objective')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                attempts += 1
                with torch.no_grad():
                    check = errors(slice(nf, None))
                    if not torch.isfinite(check).all():
                        raise FloatingPointError('Nonfinite output check')
                    if extra_ppo:
                        # 현재 PPO 기본 KL 예산 안의 마지막 후보를 저장한다.
                        anchor_kl = .5 * (((self.pi(features[nf:]) - anchor_mean[nf:]) / scales[0]) ** 2).mean()
                        valid = float(anchor_kl) <= self.config.target_kl
                    else:
                        active_check, active_before = check[list(active)], before[list(active)]
                        valid = (torch.all(active_check <= active_before * 1.01 + 1e-10)
                                 and float(active_check.sum()) < best_score)
                    if valid:
                        best_score, best_step = float(check[list(active)].sum()), step
                        best_state = [deepcopy(m.state_dict()) for m in modules]
        accepted = best_step > 0 and (extra_ppo or best_score <= (1 - MIN_REDUCTION) * score)
        for m, state in zip(modules, best_state if accepted else initial):
            m.load_state_dict(state)
        with torch.no_grad():
            after = errors(slice(nf, None))
            anchor_kl = .5 * (((self.pi(features[nf:]) - anchor_mean[nf:]) / scales[0]) ** 2).mean()
        return {'preserve_before_kl': float(before[0]), 'preserve_after_kl': float(after[0]),
                'preserve_before_value_mse': float(before[1:].sum()),
                'preserve_after_value_mse': float(after[1:].sum()),
                'preserve_before_error': float(before.sum()), 'preserve_after_error': float(after.sum()),
                'preserve_accepted': float(accepted), 'preserve_step': float(best_step if accepted else 0),
                'preserve_fit_samples': float(nf), 'preserve_check_samples': float(len(features) - nf),
                'extra_attempted_steps': float(attempts), 'extra_anchor_kl': float(anchor_kl),
                'before_reward_error': float(before[1]), 'after_reward_error': float(after[1]),
                'before_cost_error': float(before[2]), 'after_cost_error': float(after[2])}


def make_agent(method, seed, prepared, config):
    if method not in METHODS:
        raise ValueError(method)
    agent = e25_agent('fcsrl', seed, prepared, config)
    if method != 'fcsrl':
        agent.__class__ = AblatedFCSRL
        agent.mode = method
    return agent
