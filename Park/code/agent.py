"""PPO-Lag 와 FCSRL + PPO-Lag (Park 작성).

- PPOLag: 정규분포 정책 PPO 에 Lagrangian 승수를 붙인 안전 RL. 승수는 원본 FCSRL 의
  PIDLagrangianUpdater 로 갱신한다.
- FCSRL: PPOLag 의 정책과 가치망 입력에 표현 z 를 붙인다. 표현은 원본 FCSRL 의
  PPOLagReprAgent.train_encoder 를 그대로 불러 학습한다 (동역학 일관성 + feasibility 예측).

하이퍼파라미터는 팀 결과와 비교할 수 있도록 Lee/e01 과 같은 값을 쓴다 (Config 참고).
원본 FCSRL 은 커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63 을 설치해서 쓰고 고치지 않는다.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn


def _allow_fcsrl_without_safety_gymnasium():
    """원본 fcsrl 패키지는 가져올 때 safety_gymnasium 의 VisionEnvConf 이름 하나를 요구한다.

    이 실험은 Safety Gymnasium 을 쓰지 않으므로, 설치돼 있지 않으면 빈 이름만 등록한다.
    원본 코드는 고치지 않는다.
    """
    try:
        import safety_gymnasium  # noqa: F401
        return
    except ImportError:
        pass

    class VisionEnvConf:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('이 실험에서는 Safety Gymnasium 을 쓰지 않는다')

    root = types.ModuleType('safety_gymnasium')
    bases = types.ModuleType('safety_gymnasium.bases')
    underlying = types.ModuleType('safety_gymnasium.bases.underlying')
    underlying.VisionEnvConf = VisionEnvConf
    bases.underlying, root.bases = underlying, bases
    sys.modules.update({'safety_gymnasium': root, 'safety_gymnasium.bases': bases,
                        'safety_gymnasium.bases.underlying': underlying})


_allow_fcsrl_without_safety_gymnasium()
from fcsrl.agent import PPOLagReprAgent  # noqa: E402
from fcsrl.data import Batch  # noqa: E402
from fcsrl.network import Encoder, MLP  # noqa: E402
from fcsrl.utils import DiscDist, PIDLagrangianUpdater, soft_update  # noqa: E402

# 관측 표준편차의 하한. 기준 운전에서 거의 변하지 않는 관측도 정규화 뒤 너무 커지지 않게 한다
# (Yoo/e17, Lee/e01 과 같은 값).
OBS_STD_FLOORS = np.array([1., .5, 1., 1., .5, .2, 5., .1, .5, .5, .25])

# Lagrangian PID 계수. 하루 한도 15 기준 계수를 한도 72 에 맞게 환산한다 (Lee/e01 과 같은 규칙).
_PID_SCALE = 15.0 / 72.0


@dataclass(frozen=True)
class Config:
    # 수집
    n_env: int = 4
    horizon: int = 1440            # 환경 하나당 갱신마다 5일
    updates: int = 10
    # PPO
    hidden: int = 64
    init_log_std: float = -1.5
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 10
    minibatch: int = 512
    clip: float = 0.2
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    # 제약
    cost_limit: float = 72.0       # 하루 위반 스텝 한도
    max_lambda: float = 20.0
    pid: tuple = field(default=(0.02 * _PID_SCALE, 0.005 * _PID_SCALE, 0.01 * _PID_SCALE))
    # FCSRL 표현
    z_dim: int = 64
    repr_hidden: int = 256
    encoder_lr: float = 3e-4
    feasibility_discount: float = 0.9
    tau: float = 0.05              # fixed_encoder EMA
    n_buckets: int = 63
    bucket_low: float = -2.0       # symlog 공간
    bucket_high: float = 4.0


class ObsNormalizer:
    """학습 전 기본 폭기 관측으로 평균과 표준편차를 정하고 학습 중에는 바꾸지 않는다."""

    def __init__(self, observations=None):
        if observations is not None:
            observations = np.asarray(observations, dtype=float)
            self.mean = observations.mean(0)
            self.std = np.maximum(observations.std(0), OBS_STD_FLOORS)

    def __call__(self, obs):
        return np.clip((np.asarray(obs) - self.mean) / self.std, -10, 10).astype(np.float32)

    def state(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    def load(self, state):
        self.mean, self.std = np.array(state['mean']), np.array(state['std'])


def tanh_mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class PPOLag:
    name = 'ppo'

    def __init__(self, obs_dim, act_dim, config: Config, normalizer: ObsNormalizer):
        self.config = c = config
        self.normalizer = normalizer
        in_dim = obs_dim + self.extra_dim()
        self.pi = tanh_mlp(in_dim, act_dim, c.hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), c.init_log_std))
        self.vr = tanh_mlp(in_dim, 1, c.hidden)   # 보상 가치
        self.vc = tanh_mlp(in_dim, 1, c.hidden)   # 비용 가치
        self.opt_pi = torch.optim.Adam(list(self.pi.parameters()) + [self.log_std], lr=c.actor_lr)
        self.opt_v = torch.optim.Adam(list(self.vr.parameters()) + list(self.vc.parameters()), lr=c.critic_lr)
        self.pid = PIDLagrangianUpdater(0.0, list(c.pid), c.max_lambda)
        self.lagrangian = 0.0

    # 하위 클래스가 입력에 붙이는 차원
    def extra_dim(self):
        return 0

    def features(self, o):
        return o

    @staticmethod
    def tensor(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def dist(self, o):
        return torch.distributions.Normal(self.pi(self.features(o)), self.log_std.exp())

    def values(self, o):
        f = self.features(o)
        return self.vr(f)[:, 0], self.vc(f)[:, 0]

    @torch.no_grad()
    def act_det(self, obs):
        """평가용 결정적 행동. 원래 관측을 받아 [-1, 1] 행동을 낸다."""
        o = self.tensor(self.normalizer(np.atleast_2d(obs)))
        return self.pi(self.features(o)).clamp(-1, 1).numpy()[0]

    def update_lagrangian(self, mean_daily_cost):
        self.pid.update(mean_daily_cost, self.config.cost_limit)
        self.lagrangian = self.pid.get_lagrg()

    def train_aux(self, buf, idx):
        return {}

    def after_update(self):
        pass

    def update(self, buf):
        """buf: 1차원으로 편 수집 자료 (obs, act, logp, adv_r, adv_c, ret_r, ret_c, ...)."""
        c = self.config
        o, a, logp_old = self.tensor(buf['obs']), self.tensor(buf['act']), self.tensor(buf['logp'])
        ar = (buf['adv_r'] - buf['adv_r'].mean()) / (buf['adv_r'].std() + 1e-8)
        ac = buf['adv_c'] - buf['adv_c'].mean()
        lam = self.lagrangian
        adv = self.tensor((ar - lam * ac) / (1 + lam))
        ret_r, ret_c = self.tensor(buf['ret_r']), self.tensor(buf['ret_c'])
        n = len(o)
        aux_log = {}
        for epoch in range(c.epochs):
            perm = torch.randperm(n)
            kls = []
            for start in range(0, n, c.minibatch):
                idx = perm[start:start + c.minibatch]
                for k, v in self.train_aux(buf, idx).items():
                    aux_log.setdefault(k, []).append(v)
                d = self.dist(o[idx])
                logp = d.log_prob(a[idx]).sum(-1)
                ratio = torch.exp((logp - logp_old[idx]).clamp(-20, 20))
                loss_pi = -torch.min(ratio * adv[idx], ratio.clamp(1 - c.clip, 1 + c.clip) * adv[idx]).mean()
                self.opt_pi.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + [self.log_std], c.max_grad_norm)
                self.opt_pi.step()
                vr, vc = self.values(o[idx])
                loss_v = ((vr - ret_r[idx]) ** 2).mean() + ((vc - ret_c[idx]) ** 2).mean()
                self.opt_v.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(list(self.vr.parameters()) + list(self.vc.parameters()), c.max_grad_norm)
                self.opt_v.step()
                kls.append((logp_old[idx] - logp).mean().item())
            if np.mean(kls) > c.target_kl:
                break
        self.after_update()
        log = {k: float(np.mean(v)) for k, v in aux_log.items()}
        log.update(kl=float(np.mean(kls)), epochs=epoch + 1, std=float(self.log_std.detach().exp().mean()))
        return log

    def modules(self):
        return {'pi': self.pi, 'vr': self.vr, 'vc': self.vc}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: m.state_dict() for k, m in self.modules().items()}
        state['log_std'] = self.log_std.detach().clone()
        torch.save(state, path / 'model.pt')
        (path / 'normalizer.json').write_text(json.dumps(self.normalizer.state()))
        (path / 'config.json').write_text(json.dumps({'method': self.name, **asdict(self.config)}, indent=2))

    def load(self, path):
        path = Path(path)
        state = torch.load(path / 'model.pt', weights_only=True)
        for k, m in self.modules().items():
            m.load_state_dict(state[k])
        with torch.no_grad():
            self.log_std.copy_(state['log_std'])
        self.normalizer.load(json.loads((path / 'normalizer.json').read_text()))


class FCSRL(PPOLag):
    """FCSRL + PPO-Lag. 원본 train_encoder 가 쓰는 속성 이름(encoder, fixed_encoder,
    feasi_head, proj_layer, post_proj, encoder_optim, n_buckets, bucket_low, bucket_high)을 그대로 둔다."""
    name = 'fcsrl'

    def __init__(self, obs_dim, act_dim, config: Config, normalizer: ObsNormalizer):
        c = config
        hidden = [c.repr_hidden, c.repr_hidden]
        self.encoder = Encoder(obs_dim, act_dim, c.z_dim, c.z_dim, hidden)
        self.fixed_encoder = deepcopy(self.encoder).requires_grad_(False).eval()
        self.feasi_head = MLP(c.z_dim, hidden, c.n_buckets)
        self.proj_layer = MLP(c.z_dim, hidden, c.z_dim // 2)
        self.post_proj = nn.Linear(c.z_dim // 2, c.z_dim // 2, bias=False)
        params = [p for m in (self.encoder, self.feasi_head, self.proj_layer, self.post_proj) for p in m.parameters()]
        self.encoder_optim = torch.optim.Adam(params, lr=c.encoder_lr)
        self.n_buckets, self.bucket_low, self.bucket_high = c.n_buckets, c.bucket_low, c.bucket_high
        super().__init__(obs_dim, act_dim, config, normalizer)

    def extra_dim(self):
        return self.config.z_dim

    def features(self, o):
        with torch.no_grad():
            z = self.fixed_encoder.zs(o)
        return torch.cat((o, z), dim=-1)

    def _mean(self, logits):
        return DiscDist(logits.unsqueeze(0), self.bucket_low, self.bucket_high, self.n_buckets).mean().reshape(-1)

    @torch.no_grad()
    def feasibility_state(self, o):
        """F(s). 정규화한 관측을 받는다. 학습 목표의 bootstrap 에 쓴다 (원본과 같은 방식)."""
        return self._mean(self.feasi_head(self.fixed_encoder.zs(self.tensor(o)))).numpy()

    @torch.no_grad()
    def feasibility(self, o, a):
        """F(s, a). 정규화한 관측과 실제로 넣을 행동([-1, 1]) 을 받는다.

        원본 train_encoder 는 feasi_head 를 zsa(zs(s), a) 위에서 학습하므로, 행동을 넣은
        이 값이 학습된 그대로의 예측이다. 안전 차단(shield.py) 이 이 값을 쓴다.
        """
        o, a = self.tensor(np.atleast_2d(o)), self.tensor(np.atleast_2d(a))
        zs = self.fixed_encoder.zs(o)
        return self._mean(self.feasi_head(self.fixed_encoder.zsa(zs, a))).numpy()

    def train_aux(self, buf, idx):
        i = idx.numpy()
        batch = Batch(obs=self.tensor(buf['obs'][i]), obs_next=self.tensor(buf['obs_next'][i]),
                      act=self.tensor(buf['act_exec'][i]),
                      feasi_score=self.tensor(buf['feasi'][i]).unsqueeze(-1))
        return PPOLagReprAgent.train_encoder(self, batch)

    @torch.no_grad()
    def after_update(self):
        soft_update(self.fixed_encoder, self.encoder, self.config.tau)

    def modules(self):
        return {**super().modules(), 'encoder': self.encoder, 'fixed_encoder': self.fixed_encoder,
                'feasi_head': self.feasi_head, 'proj_layer': self.proj_layer, 'post_proj': self.post_proj}


METHODS = {'ppo': PPOLag, 'fcsrl': FCSRL}


def feasibility_targets(cost, end, terminated, bootstrap, discount):
    """F 학습 목표. (T, N) 배열을 뒤에서부터 계산한다 (원본 _get_feasibility 와 같은 식).

    F_t = max(c_t, discount x (1 - terminated_t) x F_{t+1})
    구간이 끝나는 스텝(end) 이나 마지막 스텝에서는 F_{t+1} 대신 bootstrap (= F(s_{t+1})) 을 쓴다.
    """
    out = np.zeros_like(cost)
    for t in reversed(range(len(cost))):
        nxt = bootstrap[t] if t == len(cost) - 1 else np.where(end[t], bootstrap[t], out[t + 1])
        out[t] = np.maximum(cost[t], discount * (1 - terminated[t]) * nxt)
    return out


def gae(rew, val, next_val, end, gamma, lam):
    """(T, N) 배열의 GAE. end 에서 사슬을 끊는다. next_val 은 종료 스텝에서 0 이어야 한다."""
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], dtype=rew.dtype)
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * next_val[t] - val[t]
        last = delta + gamma * lam * (1 - end[t]) * last
        adv[t] = last
    return adv, adv + val
