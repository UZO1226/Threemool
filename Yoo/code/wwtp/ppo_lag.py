"""PPO Lag (계획서의 PPO 비교군, Yoo/e03)

원본 FCSRL 저장소의 PPO Lag 는 Safety Gymnasium 용 설정이라 이 환경에서는 제약을 배우지 못했다
(보관 로그에서 비용 한도 초과와 seed 1의 NaN 종료를 확인했다. e03 README 참고).
그래서 OmniSafe PPOLag와 같은 이점 결합식에 원본 FCSRL의 PID 갱신기를 결합해 따로 구현했다.

- 정책: 은닉층 64 x 2 tanh MLP 가 평균을 내고, 표준편차는 상태와 무관한 학습 파라미터 (초기 log std -0.5)
- 행동은 정규분포에서 뽑은 값을 저장하고 환경에 넣을 때만 [-1, 1] 로 자른다 (확률 계산은 자르기 전 값으로)
- 보상과 비용 각각 가치망, GAE (gamma 0.99, lambda 0.95). 시간 제한으로 끝난 날은 마지막 상태 가치로 이어 붙인다
- 정책 손실의 이점: (A_r - lambda A_c) / (1 + lambda). A_r 는 정규화, A_c 는 평균만 뺀다
- Lagrange 승수: TD3 Lag 와 같은 PID 갱신기 (KP 0.02, KI 0.005, KD 0.01), 상한 20, 갱신마다 그동안 끝난 날의 평균 비용으로
- 갱신마다 병렬 환경 8개 x 1148 스텝(4일) = 9184 전이, 10 epoch, 미니배치 512, clip 0.2, 학습률 3e-4, KL 0.02 넘으면 조기 종료
- 전체 환경 스텝 목표는 TD3 쪽과 같은 약 2M (실제 2,002,112). 25 구간으로 나눠 구간마다 val 10 에피소드로 시험하고 원본 트레이너와 같은 규칙으로 저장한다
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from env import AerationEnv
from ppo_config import PPOConfig


class RunningNorm:
    def __init__(self, dim):
        self.n, self.mean, self.m2 = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x):
        x = np.atleast_2d(x)
        b = len(x)
        bm, bv = x.mean(0), x.var(0)
        d = bm - self.mean
        tot = self.n + b
        self.mean = self.mean + d * b / tot
        self.m2 = self.m2 + bv * b + d ** 2 * self.n * b / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.m2 / self.n + 1e-8), -10, 10)

    def state(self):
        return {"n": self.n, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    def load(self, s):
        self.n, self.mean, self.m2 = s["n"], np.array(s["mean"]), np.array(s["m2"])


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class PPOLagAgent:
    def __init__(self, s_dim, a_dim, device="cpu", cost_limit=15.0, max_lambda=20.0, config=None):
        from fcsrl.utils import PIDLagrangianUpdater

        self.config = c = config or PPOConfig(cost_limit=cost_limit, max_lambda=max_lambda)
        self.dev = torch.device(device)
        self.pi = mlp(s_dim, a_dim, c.hidden_dim).to(self.dev)
        self.log_std = nn.Parameter(torch.full((a_dim,), c.init_log_std, device=self.dev))
        self.vr, self.vc = mlp(s_dim, 1, c.hidden_dim).to(self.dev), mlp(s_dim, 1, c.hidden_dim).to(self.dev)
        self.opt_pi = torch.optim.Adam(list(self.pi.parameters()) + [self.log_std], lr=c.actor_lr)
        self.opt_v = torch.optim.Adam(list(self.vr.parameters()) + list(self.vc.parameters()), lr=c.critic_lr)
        self.obs_normalizer = RunningNorm(s_dim)
        self.cost_limit = c.cost_limit
        self.lagrg = 0.0
        self.pid = PIDLagrangianUpdater(0.0, [c.pid_kp, c.pid_ki, c.pid_kd], c.max_lambda)

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.dev)

    def dist(self, o):
        return torch.distributions.Normal(self.pi(self.policy_input(o)), self.log_std.exp())

    def policy_input(self, o):
        return o

    def values(self, o):
        features = self.policy_input(o)
        return self.vr(features)[:, 0], self.vc(features)[:, 0]

    def update_aux(self, buf, idx):
        return {}

    def finish_update(self):
        pass

    def adjust_advantage(self, advantage, buf):
        return advantage

    @torch.no_grad()
    def act_det(self, obs):
        """평가용 결정적 행동: 정규화한 관측 -> 평균 행동 (N, a_dim), [-1, 1]"""
        o = self._t(self.obs_normalizer(np.atleast_2d(obs)))
        return self.pi(self.policy_input(o)).clamp(-1, 1).cpu().numpy()

    def train(self, mode=True):
        pass

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({"pi": self.pi.state_dict(), "log_std": self.log_std.detach().cpu(), "vr": self.vr.state_dict(),
                    "vc": self.vc.state_dict()}, Path(path) / "model.pt")
        (Path(path) / "normalizer.json").write_text(json.dumps(self.obs_normalizer.state()))
        (Path(path) / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    def load(self, path):
        d = torch.load(Path(path) / "model.pt", map_location=self.dev)
        self.pi.load_state_dict(d["pi"])
        self.vr.load_state_dict(d["vr"])
        self.vc.load_state_dict(d["vc"])
        with torch.no_grad():
            self.log_std.copy_(d["log_std"].to(self.dev))
        self.obs_normalizer.load(json.loads((Path(path) / "normalizer.json").read_text()))

    def update(self, buf):
        c = self.config
        epochs, mb, clip, target_kl = c.update_epochs, c.minibatch_size, c.clip_ratio, c.target_kl
        o, a, logp_old = self._t(buf["obs"]), self._t(buf["act"]), self._t(buf["logp"])
        ar, ac = buf["adv_r"], buf["adv_c"]
        ar = (ar - ar.mean()) / (ar.std() + 1e-8)
        ac = ac - ac.mean()
        lam = self.lagrg
        adv = self._t(self.adjust_advantage((ar - lam * ac) / (1 + lam), buf))
        ret_r, ret_c = self._t(buf["ret_r"]), self._t(buf["ret_c"])
        n = len(o)
        info, aux = {}, {}
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.dev)
            kls = []
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                for key, value in self.update_aux(buf, idx).items():
                    aux.setdefault(key, []).append(value)
                d = self.dist(o[idx])
                logp = d.log_prob(a[idx]).sum(-1)
                ratio = torch.exp((logp - logp_old[idx]).clamp(-20, 20))
                loss_pi = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
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
            if np.mean(kls) > target_kl:
                break
        info.update(self.finish_update() or {})
        info.update({key: float(np.mean(values)) for key, values in aux.items()})
        info.update(kl=float(np.mean(kls)), pi_epochs=ep + 1, std=float(self.log_std.detach().exp().mean()))
        return info


def gae(rew, val, next_val, end, gamma=0.99, lam=0.95):
    """rew, val, next_val, end: (T, N). next_val[t] 는 스텝 t 뒤 실제 다음 상태의 가치다.
    날이 끝난 스텝(end=1, 시간 제한)은 마지막 상태 가치로 이어 붙이고 GAE 사슬은 끊는다"""
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], dtype=rew.dtype)
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * next_val[t] - val[t]
        last = delta + gamma * lam * (1.0 - end[t]) * last
        adv[t] = last
    return adv, adv + val


def evaluate(agent, envs_fn, n_episodes):
    rews, costs = [], []
    env = envs_fn()
    for _ in range(n_episodes):
        o, _ = env.reset()
        done, R, C = False, 0.0, 0.0
        while not done:
            o, r, _, tr, info = env.step(agent.act_det(o)[0])
            R += r
            C += info["cost"]
            done = tr
        rews.append(R)
        costs.append(C)
    return float(np.mean(rews)), float(np.mean(costs))


def train(data, surrogate, seed, run_dir: Path, device="cpu", config=None):
    c = config or PPOConfig()
    n_env, horizon, n_chunks = c.n_env, c.horizon, c.n_chunks
    test_episodes, cost_limit = c.test_episodes, c.cost_limit
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "effective_config.json").write_text(json.dumps({
        "method": "ppolag", "seed": seed, "device": str(device),
        "data": str(Path(data).resolve()), "surrogate": str(Path(surrogate).resolve()),
        "config": c.to_dict(), "actual_steps": c.actual_steps, "n_updates": c.n_updates,
    }, indent=2))
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = 1000 * seed
    envs = [AerationEnv(data, surrogate, split="train", seed=base + i) for i in range(n_env)]
    val_env = AerationEnv(data, surrogate, split="val", seed=base + 500)
    agent = PPOLagAgent(11, 1, device, config=c)
    obs = np.stack([e.reset()[0] for e in envs])
    agent.obs_normalizer.update(obs)
    ep_r, ep_c = np.zeros(n_env), np.zeros(n_env)
    n_updates = c.n_updates
    upc = n_updates / n_chunks  # 한 구간의 갱신 수
    best = {"epoch": -1, "reward": -np.inf, "cost": 0.0}
    log = []
    pbar, chunk, done_eps = None, 0, []
    jr = jc = 0.0
    for u in range(n_updates):
        k = int(u // upc) + 1
        if k != chunk:
            if pbar is not None:
                pbar.close()
            chunk = k
            pbar = tqdm(total=2500, desc=f"Epoch #{k}")
        B = {x: np.zeros((horizon, n_env) + s, np.float32) for x, s in
             (("obs", (11,)), ("act", (1,)), ("logp", ()), ("rew", ()), ("cost", ()), ("vr", ()), ("vc", ()),
              ("nvr", ()), ("nvc", ()), ("end", ()))}
        for t in range(horizon):
            on = agent.obs_normalizer(obs)
            with torch.no_grad():
                ot = agent._t(on)
                d = agent.dist(ot)
                a = d.sample()
                B["logp"][t] = d.log_prob(a).sum(-1).cpu().numpy()
                B["vr"][t], B["vc"][t] = agent.vr(ot)[:, 0].cpu().numpy(), agent.vc(ot)[:, 0].cpu().numpy()
            a = a.cpu().numpy()
            B["obs"][t], B["act"][t] = on, a
            nxt = np.zeros_like(obs)
            for i, e in enumerate(envs):
                o2, r, _, tr, info = e.step(np.clip(a[i], -1, 1))
                B["rew"][t, i], B["cost"][t, i] = r, info["cost"]
                ep_r[i] += r
                ep_c[i] += info["cost"]
                B["end"][t, i] = float(tr)
                nxt[i] = o2  # 날이 끝난 경우 마지막 관측 (가치를 이어 붙일 상태)
                if tr:
                    done_eps.append((ep_r[i], ep_c[i]))
                    ep_r[i] = ep_c[i] = 0.0
                    o2, _ = e.reset()
                obs[i] = o2
            with torch.no_grad():
                nt = agent._t(agent.obs_normalizer(nxt))
                B["nvr"][t], B["nvc"][t] = agent.vr(nt)[:, 0].cpu().numpy(), agent.vc(nt)[:, 0].cpu().numpy()
            agent.obs_normalizer.update(obs)
        adv_r, ret_r = gae(B["rew"], B["vr"], B["nvr"], B["end"], c.gamma, c.gae_lambda)
        adv_c, ret_c = gae(B["cost"], B["vc"], B["nvc"], B["end"], c.gamma, c.gae_lambda)
        flat = lambda x: x.reshape(horizon * n_env, *x.shape[2:])
        info = agent.update({"obs": flat(B["obs"]), "act": flat(B["act"]), "logp": flat(B["logp"]),
                             "adv_r": flat(adv_r), "adv_c": flat(adv_c), "ret_r": flat(ret_r), "ret_c": flat(ret_c)})
        if done_eps:
            jc = float(np.mean([c for _, c in done_eps]))
            jr = float(np.mean([r for r, _ in done_eps]))
            agent.pid.update(jc, cost_limit)
            agent.lagrg = agent.pid.get_lagrg()
            done_eps = []
        pbar.update(max(0, min(2500, int(round(2500 * (u + 1 - (k - 1) * upc) / upc))) - pbar.n))
        pbar.set_postfix(reward=f"{jr:.2f}", cost=f"{jc:.2f}", lam=f"{agent.lagrg:.2f}", std=f"{info['std']:.2f}")
        last_in_chunk = int((u + 1) // upc) + 1 != k or u == n_updates - 1
        if last_in_chunk:
            pbar.n = 2500
            pbar.refresh()
            tr_, tc = evaluate(agent, lambda: val_env, test_episodes)
            print(f"\nEpoch #{k} ================\ntest_reward: {tr_:.3f}\ntest_cost: {tc:.3f}", flush=True)
            if best["epoch"] == -1 or best["cost"] > cost_limit or (tr_ > best["reward"] and tc <= cost_limit):
                best = {"epoch": k, "reward": tr_, "cost": tc}
                agent.save(run_dir / "model" / f"epoch_{k}")
            log.append({"epoch": k, "test_reward": tr_, "test_cost": tc, "lambda": agent.lagrg, **info})
    pbar.close()
    agent.save(run_dir / "model_final")
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    return agent, best
