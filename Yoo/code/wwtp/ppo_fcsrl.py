"""안정화 PPO에 원본 PPO-FCSRL 표현 학습을 결합한다.

원본 czp16/FCSRL d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0.
train_encoder는 원본 메서드를 직접 호출한다. FCSRL_LICENSE 참고.
원본 PPO처럼 한 단계 표현 예측, 손실 가중치 0.5와 1, rollout 후 EMA를 사용한다.
정책과 가치망은 기존 PPO의 tanh MLP를 유지하고 입력에 고정 encoder의 z를 붙인다.
이는 논문 실험의 모든 설정을 그대로 재현하는 코드가 아닌 폐수처리 적용이다.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fcsrl.agent import PPOLagReprAgent
from fcsrl.data import Batch
from fcsrl.network import Encoder, MLP
from fcsrl.utils import DiscDist, soft_update
from ppo_lag import PPOLagAgent, mlp


@dataclass(frozen=True)
class RepresentationConfig:
    z_dim: int = 64
    hidden_dim: int = 256
    encoder_lr: float = 3e-4
    feasibility_discount: float = 0.9
    tau: float = 0.05
    n_buckets: int = 63
    bucket_low: float = -2.0
    bucket_high: float = 4.0
    frozen: bool = False

    def __post_init__(self):
        for name in ('z_dim', 'hidden_dim', 'n_buckets'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f'{name} must be an integer >= 2')
        for name in ('encoder_lr', 'feasibility_discount', 'tau', 'bucket_low', 'bucket_high'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if not 0 < self.tau <= 1 or not 0 < self.feasibility_discount <= 1:
            raise ValueError('Invalid tau or feasibility discount')
        if self.encoder_lr <= 0 or self.bucket_low >= self.bucket_high or type(self.frozen) is not bool:
            raise ValueError('Invalid representation configuration')


def feasibility_targets(cost, next_feasibility, end, terminated, discount=0.9):
    """원본 PPO의 역방향 max 재귀를 환경별로 계산한다. reset 경계를 넘지 않는다."""
    out = np.zeros_like(cost)
    for t in reversed(range(len(cost))):
        following = next_feasibility[t] if t == len(cost) - 1 else np.where(
            end[t], next_feasibility[t], out[t + 1])
        out[t] = np.maximum(cost[t], discount * (1 - terminated[t]) * following)
    return out


class PPOFCSRLAgent(PPOLagAgent):
    def __init__(self, s_dim, a_dim, device='cpu', config=None, repr_config=None):
        super().__init__(s_dim, a_dim, device, config=config)
        self.repr_config = r = repr_config or RepresentationConfig()
        c = self.config
        self.n_buckets, self.bucket_low, self.bucket_high = r.n_buckets, r.bucket_low, r.bucket_high
        hidden = [r.hidden_dim, r.hidden_dim]
        self.encoder = Encoder(s_dim, a_dim, r.z_dim, r.z_dim, hidden).to(self.dev)
        self.fixed_encoder = deepcopy(self.encoder).requires_grad_(False).eval()
        self.feasi_head = MLP(r.z_dim, hidden, r.n_buckets).to(self.dev)
        self.proj_layer = MLP(r.z_dim, hidden, r.z_dim // 2).to(self.dev)
        self.post_proj = nn.Linear(r.z_dim // 2, r.z_dim // 2, bias=False).to(self.dev)
        params = [p for module in (self.encoder, self.feasi_head, self.proj_layer, self.post_proj)
                  for p in module.parameters()]
        self.encoder_optim = torch.optim.Adam(params, lr=r.encoder_lr)
        self.pi = mlp(s_dim + r.z_dim, a_dim, c.hidden_dim).to(self.dev)
        self.vr = mlp(s_dim + r.z_dim, 1, c.hidden_dim).to(self.dev)
        self.vc = mlp(s_dim + r.z_dim, 1, c.hidden_dim).to(self.dev)
        self.opt_pi = torch.optim.Adam(list(self.pi.parameters()) + [self.log_std], lr=c.actor_lr)
        self.opt_v = torch.optim.Adam(list(self.vr.parameters()) + list(self.vc.parameters()), lr=c.critic_lr)

    def policy_input(self, o):
        with torch.no_grad():
            z = self.fixed_encoder.zs(o)
        return torch.cat((o, z), dim=-1)

    @torch.no_grad()
    def next_feasibility(self, o):
        logits = self.feasi_head(self.fixed_encoder.zs(self._t(o)))
        return DiscDist(logits, self.bucket_low, self.bucket_high, self.n_buckets).mean().cpu().numpy().reshape(-1)

    def update_aux(self, buf, idx):
        if self.repr_config.frozen:
            return {}
        indices = idx.cpu().numpy()
        batch = Batch(obs=self._t(buf['obs'][indices]), obs_next=self._t(buf['obs_next'][indices]),
                      act=self._t(buf['act_exec'][indices]),
                      feasi_score=self._t(buf['feasi'][indices]).unsqueeze(-1))
        return PPOLagReprAgent.train_encoder(self, batch)

    def update(self, buf):
        self._sync_probe = self._t(buf['obs'][:1024])
        try:
            return super().update(buf)
        finally:
            del self._sync_probe

    @torch.no_grad()
    def finish_update(self):
        if not self.repr_config.frozen:
            probe = getattr(self, '_sync_probe', None)
            before = self.dist(probe) if probe is not None else None
            soft_update(self.fixed_encoder, self.encoder, self.repr_config.tau)
            if before is not None:
                after = self.dist(probe)
                return {'encoder_sync_kl': float(torch.distributions.kl_divergence(before, after).sum(-1).mean())}
        return {}

    def save(self, path):
        super().save(path)
        path = Path(path)
        torch.save({name: getattr(self, name).state_dict() for name in
                    ('encoder', 'fixed_encoder', 'feasi_head', 'proj_layer', 'post_proj')}, path / 'representation.pt')
        (path / 'representation.json').write_text(json.dumps(asdict(self.repr_config), indent=2))

    def load(self, path):
        super().load(path)
        state = torch.load(Path(path) / 'representation.pt', map_location=self.dev, weights_only=True)
        for name, values in state.items():
            getattr(self, name).load_state_dict(values)
