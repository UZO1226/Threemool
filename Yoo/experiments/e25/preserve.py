"""원본 FCSRL EMA 뒤 정책과 가치 출력을 직전 판단에 맞춰 보정한다."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e23'))
from behavior import PriorFCSRL, make_agent as base_agent
from fcsrl.utils import soft_update

STEPS = 50
FIT_SIZE = 2048
CHECK_SIZE = 1024
MIN_REDUCTION = .2


class PreservingFCSRL(PriorFCSRL):
    def update(self, buf):
        # 독립 RNG로 rollout 전체에서 고정 방식 분할. 학습 및 행동 난수열은 보존한다.
        ids = np.random.default_rng(25000).permutation(len(buf['obs']))
        nf = min(FIT_SIZE, len(ids) * 2 // 3)
        nc = min(CHECK_SIZE, len(ids) - nf)
        self._preserve_obs = self._t(buf['obs'][ids[:nf + nc]])
        self._fit_size = nf
        try:
            return super().update(buf)
        finally:
            del self._preserve_obs, self._fit_size

    def finish_update(self):
        obs = self._preserve_obs
        nf = self._fit_size
        modules = (self.pi, self.vr, self.vc)
        with torch.no_grad():
            old_features = self.policy_input(obs)
            teachers = [net(old_features).detach() for net in modules]
            # 정책 KL과 가치 표준화 MSE. scale은 보정 fit 상태에서만 정한다.
            scales = [self.log_std.exp().detach(),
                      teachers[1][:nf].std(unbiased=False).clamp_min(1.),
                      teachers[2][:nf].std(unbiased=False).clamp_min(1.)]
            soft_update(self.fixed_encoder, self.encoder, self.repr_config.tau)
            new_features = self.policy_input(obs).detach()
        metrics = self.compensate(new_features, teachers, scales, nf)
        metrics['encoder_sync_kl'] = metrics['preserve_after_kl']
        return metrics

    def compensate(self, features, teachers, scales, nf):
        modules = (self.pi, self.vr, self.vc)
        initial = [deepcopy(m.state_dict()) for m in modules]

        def errors(sl):
            values = [(((net(features[sl]) - target[sl]) / scale) ** 2).mean()
                      for net, target, scale in zip(modules, teachers, scales)]
            values[0] = .5 * values[0]
            return torch.stack(values)

        with torch.no_grad():
            before = errors(slice(nf, None))
        best_score, best_step, best_state = float(before.sum()), 0, initial
        if best_score > 1e-12:
            # 보정 optimizer는 매 EMA마다 새로 생성한다. PPO Adam 상태는 유지한다.
            optimizer = torch.optim.Adam([
                {'params': self.pi.parameters(), 'lr': self.config.actor_lr},
                {'params': list(self.vr.parameters()) + list(self.vc.parameters()), 'lr': self.config.critic_lr}])
            for step in range(1, STEPS + 1):
                loss = errors(slice(0, nf)).sum()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([p for m in modules for p in m.parameters()], self.config.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    check = errors(slice(nf, None))
                    if torch.isfinite(check).all() and torch.all(check <= before * 1.01 + 1e-10) and float(check.sum()) < best_score:
                        best_score, best_step = float(check.sum()), step
                        best_state = [deepcopy(m.state_dict()) for m in modules]
        accepted = best_step > 0 and best_score <= (1 - MIN_REDUCTION) * float(before.sum())
        for module, state in zip(modules, best_state if accepted else initial):
            module.load_state_dict(state)
        with torch.no_grad():
            after = errors(slice(nf, None))
        return {'preserve_before_kl': float(before[0]), 'preserve_after_kl': float(after[0]),
                'preserve_before_value_mse': float(before[1:].sum()),
                'preserve_after_value_mse': float(after[1:].sum()),
                'preserve_before_error': float(before.sum()), 'preserve_after_error': float(after.sum()),
                'preserve_accepted': float(accepted), 'preserve_step': float(best_step if accepted else 0),
                'preserve_fit_samples': float(nf), 'preserve_check_samples': float(len(features) - nf)}


def make_agent(method, seed, prepared, config):
    agent = base_agent('fcsrl' if method.startswith('fcsrl') else 'ppo', seed, prepared, config)
    if method == 'fcsrl_preserve':
        agent.__class__ = PreservingFCSRL
    return agent
