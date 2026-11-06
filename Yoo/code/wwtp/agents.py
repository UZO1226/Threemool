"""방법별 에이전트 만들기와 불러오기, 평가용 행동 함수

UCB 클래스의 학습 메서드는 czp16/FCSRL의 td3_lag_repr.py에서 수정했다.
원본 커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0 (FCSRL_LICENSE).

방법(--method):
- fcsrl     : 원본 FCSRL + TD3 Lag
- td3lag    : 같은 신경망에서 표현 학습(encoder 학습)만 끈 TD3 Lag
- td3       : td3lag 에서 Lagrangian 까지 끈 제약 없는 TD3 (lambda = 0)
- ppolag    : 계획서의 PPO 비교군. ppo_lag.py 의 자체 구현 (원본 PPO Lag 는 이 환경에서 학습되지 않았다)
- fcsrl_ucb : FCSRL + TD3 Lag 의 비용 critic 을 UCB 로 바꾼 확장 (Yoo/e03)

fcsrl_ucb 는 CAL (Wu 외, "Off-Policy Primal-Dual Safe Reinforcement Learning", ICLR 2024,
https://arxiv.org/abs/2401.14758) 의 보수적 정책 최적화를 FCSRL 에 옮긴 것이다.
- 비용 critic 을 2개에서 n 개(기본 5) 앙상블로 늘린다
- 비용 critic 의 목표값과 actor 손실에서 비용 Q 를 "앙상블 평균 + k x 표준편차" (k 기본 0.5) 로 쓴다
  원본은 목표값에 2개 중 최대값, actor 손실에 평균을 쓴다
Off-policy Lagrangian 방법은 비용 Q 를 낮게 추정하는 경향이 있어 정책이 한도를 넘기 쉽다는 것이 CAL 의 지적이다.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
TD3_METHODS = ("fcsrl", "td3lag", "td3", "fcsrl_ucb")


def _ucb_agent_cls():
    from fcsrl.agent import TD3LagReprAgent
    from fcsrl.network import EnsembleEncodedCritic
    from fcsrl.utils import DeviceConfig, to_tensor

    class TD3LagReprUCBAgent(TD3LagReprAgent):
        def __init__(self, config, obs_normalizer, ucb_n=5, ucb_k=0.5):
            self.ucb_n, self.ucb_k = ucb_n, ucb_k
            super().__init__(config, obs_normalizer=obs_normalizer)

        def build_nets(self, config):
            super().build_nets(config)
            net = config.network
            self.cost_critic = EnsembleEncodedCritic(
                self.ucb_n, net.s_dim, net.a_dim, net.z_dim, net.c_critic_hidden_dim).to(DeviceConfig.DEVICE)
            self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=net.critic_lr)
            self.cost_critic_old = deepcopy(self.cost_critic)
            self.cost_critic_old.train(False)

        def _ucb(self, q):  # q: (B, n)
            return q.mean(1, keepdim=True) + self.ucb_k * q.std(1, keepdim=True)

        def _next_q(self, batch, value_type="rew"):
            if value_type == "rew":
                return super()._next_q(batch, value_type)
            with torch.no_grad():
                zs_next = self.fixed_encoder.zs(batch.obs_next)
                a_next = self.actor_old(batch.obs_next, zs_next)
                a_noise = torch.randn(size=a_next.shape, device=a_next.device) * self._policy_noise
                if self._noise_clip > 0.0:
                    a_noise = a_noise.clamp(-self._noise_clip, self._noise_clip)
                a_next = (a_next + a_noise).clamp(-1.0, 1.0)
                zsa_next = self.fixed_encoder.zsa(zs_next, a_next)
                return self._ucb(self.cost_critic_old(batch.obs_next, zs_next, a_next, zsa_next))

        def train_actor_critic(self, batch):
            """원본 train_actor_critic 과 같고, actor 손실의 비용 Q 만 UCB 로 바꿨다"""
            metrics = {}
            weight = to_tensor(getattr(batch, "weight", 1.0))
            target_q_r = to_tensor(batch.return_r)
            target_q_c = to_tensor(batch.return_c)
            with torch.no_grad():
                fixed_zs = self.fixed_encoder.zs(batch.obs)
                fixed_zsa = self.fixed_encoder.zsa(fixed_zs, batch.act)
            current_q_r = self.critic(batch.obs, fixed_zs, batch.act, fixed_zsa)
            current_q_c = self.cost_critic(batch.obs, fixed_zs, batch.act, fixed_zsa)

            critic_loss = ((current_q_r - target_q_r).pow(2).sum(-1) * weight).mean()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            metrics["loss/critic_r"] = critic_loss.item()

            cost_critic_loss = ((current_q_c - target_q_c).pow(2).sum(-1) * weight).mean()
            self.cost_critic_optim.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic_optim.step()
            metrics["loss/critic_c"] = cost_critic_loss.item()

            if self._cnt % self._update_actor_freq == 0:
                act = self.actor(batch.obs, fixed_zs) * self._act_scale + self._act_bias
                zsa = self.fixed_encoder.zsa(fixed_zs, act)
                q_r = self.critic(batch.obs, fixed_zs, act, zsa).mean()
                q_c_all = self.cost_critic(batch.obs, fixed_zs, act, zsa)
                q_c = self._ucb(q_c_all).mean()
                self.lagrg = self.lagrg_updater.get_lagrg()
                actor_loss = -(q_r - self.lagrg * q_c)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                self._last_qr, self._last_qc = q_r.item(), q_c.item()
                metrics["loss/actor_qr"] = self._last_qr
                metrics["loss/actor_qc"] = self._last_qc
                metrics["loss/actor_qc_std"] = q_c_all.std(1).mean().item()
                self.sync_weights()
            self._cnt += 1
            return metrics

    return TD3LagReprUCBAgent


def config_path(method):
    return HERE / ("ppo_wwtp.yaml" if method == "ppolag" else "td3_wwtp.yaml")


def make_agent(method, cfg: dict, env, ucb_n=5, ucb_k=0.5):
    """cfg: yaml 을 읽은 dict (필요하면 학습 스크립트가 고친 것)"""
    if method == "fcsrl_ucb":
        if type(ucb_n) is not int or ucb_n < 2:
            raise ValueError("ucb_n must be an integer >= 2 for sample standard deviation")
        if not np.isfinite(ucb_k) or ucb_k < 0:
            raise ValueError("ucb_k must be finite and nonnegative")
    from fcsrl.utils import MeanStdNormalizer, dict2attr

    if method == "ppolag":
        from ppo_config import PPOConfig
        from ppo_lag import PPOLagAgent
        from fcsrl.utils import DeviceConfig

        return PPOLagAgent(int(np.prod(env.observation_space.shape)),
                           int(np.prod(env.action_space.shape)), DeviceConfig.DEVICE,
                           config=PPOConfig.from_dict(cfg)), cfg
    config = dict2attr(cfg)
    config.network.s_dim = int(np.prod(env.observation_space.shape))
    config.network.a_dim = int(np.prod(env.action_space.shape))
    config.agent.action_range = [float(env.action_space.low[0]), float(env.action_space.high[0])]
    norm = MeanStdNormalizer()
    if method == "fcsrl_ucb":
        agent = _ucb_agent_cls()(config, norm, ucb_n=ucb_n, ucb_k=ucb_k)
    else:
        from fcsrl.agent import TD3LagReprAgent
        agent = TD3LagReprAgent(config, obs_normalizer=norm)
    if method in ("td3lag", "td3"):
        agent.train_encoder = lambda batch: {}  # 표현 학습 끄기. encoder 는 초기값 그대로 고정된다
    return agent, config


def checkpoint_path(run_dir: Path, metrics: dict, checkpoint="best"):
    """마지막 epoch에 저장한 선택 모델도 로그로 확인되면 마지막 모델로 사용할 수 있다."""
    import re

    run_dir = Path(run_dir)
    if checkpoint not in ("best", "final"):
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    best = run_dir / "model" / f"epoch_{metrics['best_epoch']}"
    path = best if checkpoint == "best" else run_dir / "model_final"
    if checkpoint == "final" and not (path / "model.pt").exists():
        logs = list(run_dir.glob("wandb/offline-run-*/files/output.log"))
        if len(logs) == 1:
            epochs = [int(e) for e in re.findall(r"Epoch #(\d+) =+", logs[0].read_text())]
            if epochs and metrics["best_epoch"] == max(epochs):
                path = best
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"{checkpoint} checkpoint unavailable: {path}")
    return path


def load_agent(run_dir: Path, env, cudaid=0, checkpoint="best"):
    """run_dir/metrics.json 의 방법과 best_epoch 로 에이전트를 다시 만들고 가중치를 불러온다"""
    import json

    from fcsrl.utils import DeviceConfig

    DeviceConfig().select_device(cudaid)
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    method = m["method"]
    model_dir = checkpoint_path(run_dir, m, checkpoint)
    if method == "ppolag":
        from ppo_config import PPOConfig
        from ppo_lag import PPOLagAgent
        cfg_path = model_dir / "config.json"
        # 기존 e03 체크포인트는 설정 파일이 없다. 당시 실제 기본값으로 읽는다.
        cfg = PPOConfig.from_dict(json.loads(cfg_path.read_text())) if cfg_path.exists() else PPOConfig()
        agent = PPOLagAgent(int(np.prod(env.observation_space.shape)), 1, DeviceConfig.DEVICE, config=cfg)
        agent.load(model_dir)
        agent.checkpoint_path = str(model_dir)
        return agent
    cfg = yaml.safe_load(open(config_path(method)))
    agent, _ = make_agent(method, cfg, env, ucb_n=m.get("ucb_n", 5), ucb_k=m.get("ucb_k", 0.5))
    agent.load_model(str(model_dir))
    agent.train(False)
    agent.checkpoint_path = str(model_dir)
    return agent


def act_fn(agent):
    """평가용 결정적 행동 함수 obs -> action. PPO 는 정책 분포의 평균을 쓴다"""
    from fcsrl.data import Batch
    from fcsrl.agent import PPOLagReprAgent

    if hasattr(agent, "act_det"):
        return lambda obs, _env=None: agent.act_det(obs)[0]
    is_ppo = isinstance(agent, PPOLagReprAgent)

    def act(obs, _env=None):
        o = agent.obs_normalizer(obs[None])
        with torch.no_grad():
            if is_ppo:
                ot = torch.as_tensor(o, dtype=torch.float32, device=next(agent.actor.parameters()).device)
                mu, _ = agent.actor(ot, agent.fixed_encoder.zs(ot))
                a = mu.clamp(-1.0, 1.0) * agent._act_scale + agent._act_bias
            else:
                res, _ = agent(Batch(obs=o, info={}), add_noise=False)
                a = res.act
        return a.detach().cpu().numpy()[0]

    return act
