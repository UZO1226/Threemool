"""e33 알고리즘을 유지하고 두 군의 입력 용량을16으로 맞춘다."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e33'))
from dual import DualPPO, DualFCSRL, DualActor, METHODS, LIMITS, config, save
from ppo_lag import RunningNorm, mlp
from ppo_fcsrl import PPOFCSRLAgent
from fcsrl.utils import PIDLagrangianUpdater
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes, AIR_MAX
from common import sha
sys.path.insert(0, str(HERE))

MODES = ('padded', 'previous')


class CommandEnv(BatchEnv):
    def __init__(self, days, dynamics, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        super().__init__(days, dynamics)

    def obs(self):
        original = super().obs()
        extra = self.prev_zone / AIR_MAX if self.mode == 'previous' and self.t > 0 else np.zeros_like(self.prev_zone)
        return np.column_stack([original, extra]).astype(np.float32)


class PrefixNorm:
    """원11개 정규화는 그대로, 이미9000으로 나눈 추가5개는 그대로 전달한다."""
    def __init__(self, state):
        self.base = RunningNorm(11)
        self.base.load(state)

    def __call__(self, x):
        x = np.asarray(x)
        assert x.shape[-1] == 16
        return np.concatenate([self.base(x[..., :11]), x[..., 11:]], axis=-1)

    def state(self):
        return {'kind': 'prefix11_passthrough5', 'base': self.base.state()}

    def load(self, state):
        assert state['kind'] == 'prefix11_passthrough5'
        self.base.load(state['base'])


class PrefixPrior:
    def prior_action(self, normalized_obs):
        with torch.no_grad():
            return self.prior(normalized_obs[..., :11]).clamp(-1, 1)


class CommandPPO(PrefixPrior, DualPPO):
    pass


class CommandFCSRL(PrefixPrior, DualFCSRL):
    pass


class CommandActor(PrefixPrior, DualActor):
    pass


def make_agent(method, seed, prepared, c):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = {'ppo': CommandPPO, 'fcsrl': CommandFCSRL, 'fcsrl_actor': CommandActor}[method]
    a = cls(16, 1, config=c)
    if method == 'fcsrl_actor':
        a.mode = method
    a.prior = mlp(11, 1).to(a.dev)
    a.prior.load_state_dict(torch.load(prepared / 'prior.pt', map_location=a.dev, weights_only=True))
    a.prior.requires_grad_(False).eval()
    a.obs_normalizer = PrefixNorm(json.loads((prepared / 'normalizer.json').read_text()))
    with torch.no_grad():
        a.pi[-1].weight.zero_()
        a.pi[-1].bias.zero_()
    # e33 두 비용 head 초기화와 동일. 두 관측 군의 가중치/난수열도 동일하다.
    rng = torch.get_rng_state()
    old = a.vc
    a.vc = mlp(old[0].in_features, 2, c.hidden_dim).to(a.dev)
    with torch.no_grad():
        for i in (0, 2):
            a.vc[i].load_state_dict(old[i].state_dict())
        a.vc[-1].weight.copy_(old[-1].weight.repeat(2, 1) / 2)
        a.vc[-1].bias.copy_(old[-1].bias.repeat(2) / 2)
    torch.set_rng_state(rng)
    a.opt_v = torch.optim.Adam(list(a.vr.parameters()) + list(a.vc.parameters()), lr=c.critic_lr)
    a.multipliers = np.zeros(2)
    a.constraint_pids = [PIDLagrangianUpdater(0., [c.pid_kp, c.pid_ki, c.pid_kd], c.max_lambda) for _ in range(2)]
    return a
