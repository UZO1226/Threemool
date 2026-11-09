"""원본 FCSRL 손실을 유지하고 1시간 평균 위반 비용 예측을 추가한다.

train_encoder의 원본 손실 계산은 czp16/FCSRL d882b910, Apache-2.0을 따른다.
기존 Yoo/code/wwtp/FCSRL_LICENSE 참조. 추론 시 추가 head는 사용하지 않는다.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e23'))
from behavior import PriorFCSRL, make_agent as base_agent
from fcsrl.utils import DiscDist
from fcsrl.agent.ppo_lag_repr import cosine_sim_loss

WINDOW = 12
WEIGHT = 1.0


def recovery_targets(cost, end, window=WINDOW):
    """완전히 관측된 연속 window개 비용만 평균낸다. reset과 rollout 끝을 넘지 않는다."""
    cost, end = np.asarray(cost), np.asarray(end)
    if cost.shape != end.shape or cost.ndim != 2 or window < 1:
        raise ValueError('Expected matching (T,N) arrays and a positive window')
    target, valid = np.zeros_like(cost), np.zeros_like(cost, dtype=bool)
    for t in range(len(cost) - window + 1):
        valid[t] = ~np.any(end[t:t + window - 1].astype(bool), axis=0)
        target[t] = cost[t:t + window].mean(axis=0)
    return target, valid.astype(np.float32)


class RecoveryFCSRL(PriorFCSRL):
    def auxiliary_losses(self, obs, obs_next, act, feasi, recovery, mask):
        with torch.no_grad():
            target_zs = self.fixed_encoder.zs(obs_next)
            target_ps = self.proj_layer(target_zs)
        zs = self.encoder.zs(obs)
        pred_zs = self.encoder.zsa(zs, act)
        pred_ps = self.post_proj(self.proj_layer(pred_zs))
        dynamics = -cosine_sim_loss(pred_ps, target_ps).mean()
        dist = DiscDist(self.feasi_head(pred_zs).unsqueeze(0), self.bucket_low, self.bucket_high, self.n_buckets)
        feasibility = -dist.log_prob(feasi.unsqueeze(0)).mean()
        latent = pred_zs.detach() if self.detach_recovery else pred_zs
        pred_recovery = self.recovery_head(latent)[:, 0]
        recovery_loss = (((pred_recovery - recovery) ** 2) * mask).sum() / mask.sum().clamp_min(1.)
        return dynamics, feasibility, recovery_loss, dist.mean().detach().squeeze(0), pred_recovery

    def update_aux(self, buf, idx):
        ids = idx.cpu().numpy()
        args = [self._t(buf[k][ids]) for k in ('obs', 'obs_next', 'act_exec')]
        feasi = self._t(buf['feasi'][ids]).unsqueeze(-1)
        target, mask = self._t(buf['recovery'][ids]), self._t(buf['recovery_mask'][ids])
        dyn, fea, rec, mean_feasi, _ = self.auxiliary_losses(*args, feasi, target, mask)
        original_loss = .5 * dyn + fea
        total = original_loss + WEIGHT * rec
        self.encoder_optim.zero_grad()
        total.backward()
        self.encoder_optim.step()
        return {'encoder_dynamics': dyn.item(), 'encoder_pred_feasi': fea.item(),
                'encoder_pred_feasi_MSE': F.mse_loss(mean_feasi, feasi).item(),
                'encoder': total.item(), 'recovery_mse': rec.item(),
                'original_encoder_loss': original_loss.item()}

    def save(self, path):
        super().save(path)
        torch.save(self.recovery_head.state_dict(), Path(path) / 'recovery_head.pt')

    def load(self, path):
        super().load(path)
        self.recovery_head.load_state_dict(torch.load(Path(path) / 'recovery_head.pt', map_location=self.dev, weights_only=True))


def make_agent(method, seed, prepared, config):
    agent = base_agent('fcsrl' if method.startswith('fcsrl') else 'ppo', seed, prepared, config)
    if method in ('fcsrl_recovery', 'fcsrl_detached'):
        agent.__class__ = RecoveryFCSRL
        # head 추가가 rollout/미니배치 난수열과 기존 망 초기값을 바꾸지 않게 한다.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(50000 + seed)
            agent.recovery_head = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)).to(agent.dev)
        agent.detach_recovery = method == 'fcsrl_detached'
        agent.encoder_optim.add_param_group({'params': agent.recovery_head.parameters()})
    return agent
