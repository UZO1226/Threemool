"""원 FCSRL 손실에 연속 NH4 초과 크기 보조 예측을 더한다.

원 train_encoder 계산은 czp16/FCSRL d882b91, Apache-2.0을 따른다.
Yoo/e24 recovery.py의 별도 head와 경사 분리 대조 구조를 재사용한다.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e56'))
import hybrid as h
base = h.base
from behavior import PriorFCSRL
from fcsrl.utils import DiscDist
from fcsrl.agent.ppo_lag_repr import cosine_sim_loss

METHODS = ('baseline', 'severity', 'detached')
WINDOW = 4
SCALE = 4.0
WEIGHT = 1.0
if 'severity_head' not in h.MODULES:
    h.MODULES.append('severity_head')


def targets(nh4_after, end):
    """현재 행동 뒤부터 4개 전이의 초과 평균/4. 날짜 또는 buffer 경계를 넘지 않는다."""
    assert nh4_after.shape == end.shape and nh4_after.ndim == 2
    out = np.zeros_like(nh4_after, dtype=np.float32)
    mask = np.zeros_like(out)
    for t in range(len(out) - WINDOW + 1):
        out[t] = np.maximum(nh4_after[t:t+WINDOW] - 4., 0.).mean(0) / SCALE
        mask[t] = ~np.any(end[t:t+WINDOW-1].astype(bool), axis=0)
    return out, mask


class RecordingEnv(base.BatchEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nh4_after = []

    def step(self, action):
        value = super().step(action)
        self.nh4_after.append(self.state[:, 0].copy())
        return value


def collect(agent, c, env, dates, rng):
    env.nh4_after = []
    buf, diag = base.e30.collect(agent, c, env, dates, rng)
    nh4 = np.asarray(env.nh4_after)
    target, mask = targets(nh4, buf['end'].reshape(c.horizon, c.n_env))
    assert nh4.shape == (c.horizon, c.n_env)
    buf['severity'] = target.ravel()
    buf['severity_mask'] = mask.ravel()
    diag.update(severity_valid_fraction=float(mask.mean()),
                severity_target_mean=float(target[mask.astype(bool)].mean()),
                severity_positive_fraction=float((target[mask.astype(bool)] > 0).mean()))
    return buf, diag


class SeverityFCSRL(PriorFCSRL):
    def losses(self, buf, idx):
        ids = idx.cpu().numpy()
        obs, nxt, act = [self._t(buf[k][ids]) for k in ('obs', 'obs_next', 'act_exec')]
        feasi = self._t(buf['feasi'][ids]).unsqueeze(-1)
        with torch.no_grad():
            target_ps = self.proj_layer(self.fixed_encoder.zs(nxt))
        z = self.encoder.zsa(self.encoder.zs(obs), act)
        pred_ps = self.post_proj(self.proj_layer(z))
        dynamics = -cosine_sim_loss(pred_ps, target_ps).mean()
        dist = DiscDist(self.feasi_head(z).unsqueeze(0), self.bucket_low, self.bucket_high, self.n_buckets)
        feasibility = -dist.log_prob(feasi.unsqueeze(0)).mean()
        latent = z.detach() if self.detach_severity else z
        prediction = self.severity_head(latent)[:, 0]
        mask = self._t(buf['severity_mask'][ids])
        target = self._t(buf['severity'][ids])
        # UNREAL의 zero/nonzero 균형을 보조 손실에만 적용한다. PPO 표본은 바꾸지 않는다.
        loss = balanced_mse(prediction, target, mask)
        return dynamics, feasibility, loss, F.mse_loss(dist.mean().detach().squeeze(0), feasi)

    def update_aux(self, buf, idx):
        dynamics, feasibility, severity, feasi_mse = self.losses(buf, idx)
        original = .5 * dynamics + feasibility
        total = original + WEIGHT * severity
        self.encoder_optim.zero_grad()
        total.backward()
        self.encoder_optim.step()
        return dict(encoder_dynamics=dynamics.item(), encoder_pred_feasi=feasibility.item(),
                    encoder_pred_feasi_MSE=feasi_mse.item(), encoder=total.item(),
                    original_encoder_loss=original.item(), severity_mse=severity.item())

    def save(self, path):
        super().save(path)
        torch.save(self.severity_head.state_dict(), Path(path) / 'severity_head.pt')

    def load(self, path):
        super().load(path)
        self.severity_head.load_state_dict(torch.load(Path(path) / 'severity_head.pt', map_location=self.dev, weights_only=True))


def make_agent(method, seed, prepared, c):
    assert method in METHODS
    h.configure('cpu')
    agent = base.common.make_agent('fcsrl', seed, prepared, c)
    if method != 'baseline':
        agent.__class__ = SeverityFCSRL
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(58000 + seed)
            agent.severity_head = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        agent.detach_severity = method == 'detached'
        agent.encoder_optim.add_param_group({'params': agent.severity_head.parameters()})
    return agent


def make_pair(method, seed, prepared, c, device='cuda:0'):
    learner = make_agent(method, seed, prepared, c)
    collector = deepcopy(learner)
    if device != 'cpu':
        h.configure(device)
        learner.dev = torch.device(device)
        for module in h.modules(learner).values():
            module.to(device)
        learner.log_std = nn.Parameter(learner.log_std.detach().to(device))
        learner.opt_pi = torch.optim.Adam(list(learner.pi.parameters()) + [learner.log_std], lr=c.actor_lr)
        learner.opt_v = torch.optim.Adam(list(learner.vr.parameters()) + list(learner.vc.parameters()), lr=c.critic_lr)
        names = ['encoder', 'feasi_head', 'proj_layer', 'post_proj']
        if hasattr(learner, 'severity_head'):
            names.append('severity_head')
        learner.encoder_optim = torch.optim.Adam([p for n in names for p in getattr(learner, n).parameters()], lr=learner.repr_config.encoder_lr)
    return learner, collector


def fingerprint(agent):
    import hashlib
    digest = hashlib.sha256(base.fingerprint(agent).encode())
    if hasattr(agent, 'severity_head'):
        for value in agent.severity_head.state_dict().values():
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def balanced_mse(prediction, target, valid):
    terms = []
    error = (prediction - target).square()
    for selection in [(valid > 0) & (target > 0), (valid > 0) & (target == 0)]:
        if selection.any():
            terms.append(error[selection].mean())
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.


def gradient_diagnostics(agent, buf):
    if not hasattr(agent, 'severity_head'):
        return {}
    h.configure(str(agent.dev))
    idx = torch.arange(min(512, len(buf['obs'])), device=agent.dev)
    dynamics, feasibility, auxiliary, _ = agent.losses(buf, idx)
    params = list(agent.encoder.parameters())
    g = torch.autograd.grad(.5*dynamics+feasibility, params, retain_graph=True, allow_unused=True)
    a = torch.autograd.grad(auxiliary, params, allow_unused=True)
    flat = lambda xs: torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1) for x,p in zip(xs,params)])
    gv, av = flat(g), flat(a)
    norm = gv.norm()
    return dict(aux_gradient_norm_ratio=float(av.norm()/(norm+1e-12)),
                aux_gradient_cosine=float(torch.dot(gv,av)/(norm*av.norm()+1e-12)))
