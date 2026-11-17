"""e68 학습: 원본 FCSRL PPO 에이전트 (czp16/FCSRL PPOLagReprAgent) 를 그대로 쓰고 수집만 DARROW 하루 환경으로 바꾼다.

원본에서 그대로 쓰는 것: 망 구조 (EncodedActorProb, EncodedCritic, Encoder, 256 x 2), 손실, dual clip PPO, GAE 0.97,
인코더 갱신 (미니배치마다 0.5 x 동역학 + 가능성), EMA tau 0.05 (수집 한 번마다), 63 구간 가능성 머리, PID 승수
(KP 0.02, KI 0.005, KD 0.01, 상한 1.0), 정규화기, 미니배치 64, 반복 2, 시작 무작위 16 에피소드.
수집 규모는 원본 (에피소드 32개 x 250 결정 = 8,000 전이, 갱신 250번 = 약 2M) 에 맞춰 하루 84개 x 95 = 7,980 전이, 갱신 250번.

바꾼 것 (README 에 근거):
1. 하루가 끝나면 에피소드가 정말 끝난다 (다음 날은 기록 상태에서 다시 시작하고 예산도 새로 시작). 그래서 끝 스텝을
   terminate = trunc = 1 로 저장하고, 원본 _get_returns 가 trunc 에서 다음 상태 가치를 무조건 더하는 부분에
   (1 - terminate) 를 곱한다. 원본은 이 자리에서 비용 반환에도 보상 가치망을 부르는데 (value_type 누락), 여기서는
   비용 가치망을 부른다. 끝 스텝은 항상 terminate = 1 이라 두 차이 모두 값에 영향이 없다.
2. 비용: 환경 비용은 예산 초과 스텝마다 1 (env68 설명). 원본 Safety Gym 처럼 스텝당 이진 비용이며 척도를 바꾸지 않는다.
   한도는 하루 예산 초과 스텝 0.3 개로, dev 날짜 기록 운전 재생 값 0.289 (위반일 2.6 %, 위반일 평균 11.1 스텝) 와 같은
   수준이다. 가능성 목표는 원본 정의대로 이 0/1 비용으로 계산한다.
3. 평가: 원본 test_agent 처럼 표본 행동으로 평가한다 (고정 seed 3개의 평균이 주 평가). 학습된 표준편차가 1 ~ 2 라
   잘린 표본은 사실상 확률적 켜고 끄기이고, 평균 행동은 학습한 정책과 다르게 움직인다 (시험 학습에서 평균 행동 위반일
   64 ~ 74 % 대 표본 22 ~ 55 %). 평균 행동 결과는 참고로 남긴다.
4. 수치 안정: train_actor_critic 의 로그 비율을 [-20, 20] 으로 자른다 (메서드 설명 참고).
5. 행동 확률: 원본은 정규분포 표본을 [-1, 1] 로 자른 값을 저장하고 그 값의 확률로 PPO 비율을 계산한다. 이 문제는
   최적 행동 (공기 0) 이 범위 끝에 있어, 시험 학습에서 표준편차가 1 에서 1.6 으로 커지며 정책이 사실상 무작위 켜고 끄기가
   됐다. 표준 PPO 처럼 자르기 전 표본을 저장해 확률을 계산하고 환경에만 자른 값을 넣는다. 표현의 동역학 예측에는 자른
   (실제로 넣은) 행동을 쓴다. 두 방법에 똑같이 적용한다.
6. 승수 상한: 원본 1.0 대신 20 (기존 Yoo PPO-Lag 설정과 같음). 시험 학습 (dev, seed 0, 갱신 60번) 에서 상한 1.0 은
   승수가 상한에 붙은 채 하루 예산 초과 스텝이 5 ~ 8 에 머물렀고, 상한 20 은 승수 약 2 에서 2.2 ~ 2.6 까지 내려갔다.
   원본 상한은 Safety Gym 보상 척도에 맞춘 값이고, 여기서는 공기 보상이 바로 들어오고 NH4 초과 비용은 늦게 나온다.
방법:
- fcsrl: 원본 그대로.
- ppolag: 같은 에이전트에서 인코더 학습만 끈다 (train_encoder 를 부르지 않는다). 고정 인코더는 처음 무작위 가중치 그대로라
  정책 입력의 z 는 학습되지 않은 무작위 특징이다. 표준 PPO-Lag 가 아니라 "같은 망의 PPO-Lag + 고정 무작위 z" 이며,
  표현 학습의 기여만 분리한다. 같은 seed 에서 초기 망과 난수 흐름이 같아 seed 짝 비교가 된다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env68 as E  # noqa: E402

FCSRL_ROOT = Path('/home/jio/fcsrl_src/FCSRL')
sys.path.insert(0, str(FCSRL_ROOT))
from fcsrl.agent import PPOLagReprAgent  # noqa: E402
from fcsrl.data import Batch  # noqa: E402
from fcsrl.utils import MeanStdNormalizer, dict2attr, to_numpy  # noqa: E402

COST_SCALE = 1.0
COST_LIMIT = 0.3
MAX_LAMBDA = 20.0
EVAL_SAMPLE_SEEDS = (0, 1, 2)
N_EPISODES = 84
UPDATES = 250
WARMUP_EPISODES = 16
HYPER = FCSRL_ROOT / 'hyper_params/PPORepr_Lag.yaml'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_config(seed, max_lambda):
    cfg = yaml.safe_load(HYPER.read_text())
    cfg['network'].update(s_dim=E.OBS_DIM, a_dim=1, repr_type='FCSRL')
    cfg['agent']['action_range'] = [-1.0, 1.0]
    cfg['Lagrangian']['constraint_threshold'] = COST_LIMIT
    cfg['Lagrangian']['max_lambda'] = max_lambda
    cfg['misc']['seed'] = seed
    return cfg


class Agent68(PPOLagReprAgent):
    def __init__(self, cfg, learn_repr):
        super().__init__(dict2attr(cfg), obs_normalizer=MeanStdNormalizer())
        self.learn_repr = learn_repr

    def _get_returns(self, batch):
        returns = []
        for value_type in ['rew', 'cost']:
            ret = getattr(batch, value_type).copy()
            for i in reversed(range(len(ret))):
                if batch.trunc[i] or i == len(ret) - 1:
                    zs_next = self.fixed_encoder.zs(batch.obs_next[i:i + 1])
                    next_value = self._next_v(batch.obs_next[i:i + 1], zs_next, value_type)
                    ret[i] += (1 - batch.terminate[i]) * to_numpy(next_value.squeeze())
                else:
                    ret[i] += self._gamma * (1 - batch.terminate[i]) * ret[i + 1]
            returns.append(ret)
        return returns

    def _get_feasibility(self, batch):
        scaled = batch.cost
        batch.cost = (scaled > 0).astype(scaled.dtype)
        try:
            return super()._get_feasibility(batch)
        finally:
            batch.cost = scaled

    def train_encoder(self, batch):
        if not self.learn_repr:
            return {}
        raw = batch.act
        batch.act = raw.clamp(-1., 1.)  # 표현의 동역학 예측은 환경에 실제로 넣은 행동으로 배운다
        try:
            return super().train_encoder(batch)
        finally:
            batch.act = raw

    def forward(self, batch, states=None):
        """원본 forward 와 같고 자르기 전 표본을 돌려준다 (변경 5). 원본은 표본을 [-1, 1] 로 자른 뒤 그 값의 확률을 쓴다."""
        zs = self.fixed_encoder.zs(batch.obs)
        dist = D.Independent(D.Normal(*self.actor(batch.obs, zs)), 1)
        act = dist.sample()
        return Batch(act=self._act_scale * act + self._act_bias, dist=dist, log_p_act=dist.log_prob(act)), None

    def train_actor_critic(self, batch):
        """원본 train_actor_critic 를 그대로 옮기고 로그 비율만 [-20, 20] 으로 자른다 (변경 4).

        원본은 한 번의 learn 에서 KL 조기 종료 없이 미니배치 250번을 돌린다. 그 사이 정책이 많이 움직이면 [-1, 1] 끝에서
        잘린 행동의 로그 비율이 100 을 넘고, exp 가 float32 에서 inf 가 되어 torch.where 기울기가 NaN 이 된다
        (첫 실행에서 20개 중 2개가 갱신 5번 안에 이렇게 멈췄다). 비율이 1 +- 0.2 밖이면 clip 과 dual clip 때문에
        그 표본의 기울기는 원래 0 (비율이 e^-20 보다 작은 쪽은 e^-20 이하) 이므로, 자르기는 inf 만 막고 손실과 기울기는
        사실상 바꾸지 않는다. 잘린 표본 비율은 log_ratio_clamped_pct 로 남긴다.
        """
        metrics = {}
        with torch.no_grad():
            fixed_zs = self.fixed_encoder.zs(batch.obs)
        dist = D.Independent(D.Normal(*self.actor(batch.obs, fixed_zs)), 1)
        dist_old = D.Independent(D.Normal(*self.actor_old(batch.obs, fixed_zs)), 1)
        act_normed = (batch.act - self._act_bias) / self._act_scale
        log_ratio = dist.log_prob(act_normed) - dist_old.log_prob(act_normed)
        ratio = torch.exp(log_ratio.clamp(-20., 20.)).unsqueeze(-1)  # 원본: torch.exp(log_ratio)

        vf_loss = F.mse_loss(self.critic(batch.obs, fixed_zs), batch.return_r)
        self.critic_optim.zero_grad()
        vf_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self._max_grad_norm)
        self.critic_optim.step()
        metrics['r_critic'] = vf_loss.item()

        c_vf_loss = F.mse_loss(self.cost_critic(batch.obs, fixed_zs), batch.return_c)
        self.cost_critic_optim.zero_grad()
        c_vf_loss.backward()
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self._max_grad_norm)
        self.cost_critic_optim.step()
        metrics['c_critic'] = c_vf_loss.item()

        rew_surr1 = ratio * batch.adv
        rew_surr2 = ratio.clamp(1. - self._clip_ratio_eps, 1. + self._clip_ratio_eps) * batch.adv
        rew_clip1 = torch.min(rew_surr1, rew_surr2)
        rew_clip2 = torch.max(rew_clip1, self._dual_clip_ratio * batch.adv)
        rew_surr = torch.where(batch.adv < 0, rew_clip2, rew_clip1).mean()

        cost_surr1 = ratio * batch.c_adv
        cost_surr2 = ratio.clamp(1. - self._clip_ratio_eps, 1. + self._clip_ratio_eps) * batch.c_adv
        cost_clip1 = torch.max(cost_surr1, cost_surr2)
        cost_clip2 = torch.min(cost_clip1, self._dual_clip_ratio * batch.c_adv)
        cost_surr = torch.where(batch.c_adv > 0, cost_clip2, cost_clip1).mean()

        entropy_loss = dist.entropy().mean()
        KL_loss = D.kl.kl_divergence(dist, dist_old).mean()
        self.lagrg = self.lagrg_updater.get_lagrg()
        actor_loss = - (rew_surr - self.lagrg * cost_surr) - self._entropy_coef * entropy_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self._max_grad_norm)
        self.actor_optim.step()

        metrics.update({'kld': KL_loss.item(), 'entropy': entropy_loss.item(), 'r_adv': rew_surr.item(), 'c_adv': cost_surr.item(),
                        'log_ratio_clamped_pct': float(100 * (log_ratio.detach().abs() > 20).float().mean())})
        return metrics

    @torch.no_grad()
    def act_det(self, raw_obs):
        o = self.obs_normalizer(raw_obs)
        mu, _ = self.actor(o, self.fixed_encoder.zs(o))
        return to_numpy(mu.clamp(-1, 1))[:, 0]

    @torch.no_grad()
    def act_sample(self, raw_obs):
        o = self.obs_normalizer(raw_obs)
        return np.clip(to_numpy(self(Batch(obs=o))[0].act)[:, 0], -1, 1)

    @torch.no_grad()
    def feasibility(self, raw_obs):
        from fcsrl.utils import DiscDist
        o = self.obs_normalizer(raw_obs)
        logits = self.feasi_head(self.fixed_encoder.zs(o)).unsqueeze(0)
        return to_numpy(DiscDist(logits, self.bucket_low, self.bucket_high, self.n_buckets).mean()).reshape(-1)

    def state(self):
        rms = self.obs_normalizer.rms
        return dict(nets={k: getattr(self, k).state_dict() for k in ('actor', 'critic', 'cost_critic', 'encoder', 'fixed_encoder', 'feasi_head', 'proj_layer', 'post_proj')},
                    normalizer=dict(mean=rms.mean, var=rms.var, count=rms.count), lagrg=self.lagrg)

    def load_state(self, s):
        for k, v in s['nets'].items():
            getattr(self, k).load_state_dict(v)
        from fcsrl.utils.normalizer import RunningMeanStd
        rms = RunningMeanStd(shape=(1, E.OBS_DIM))
        rms.mean, rms.var, rms.count = s['normalizer']['mean'], s['normalizer']['var'], s['normalizer']['count']
        self.obs_normalizer.rms = rms
        self.obs_normalizer.set_read_only()


def collect(agent, env, dates, rng, n, random=False):
    """하루 n 개를 끝까지 돈다. 원본 Collector 처럼 학습 수집 중에만 정규화기를 갱신하고 저장은 원 관측이다."""
    eps = [(dates[i], int(m)) for i, m in zip(rng.integers(len(dates), size=n), rng.integers(env.dyn.n_members, size=n))]
    obs = env.reset(eps)
    T = env.T
    buf = {k: [] for k in ('obs', 'act', 'rew', 'cost', 'obs_next')}
    air = np.zeros(n)
    for t in range(T):
        agent.obs_normalizer.unset_read_only()
        on = agent.obs_normalizer(obs)
        agent.obs_normalizer.set_read_only()
        if random:
            act = rng.uniform(-1, 1, size=(n, 1))
        else:
            with torch.no_grad():
                act = to_numpy(agent(Batch(obs=on))[0].act)
        nxt, rew, cost, done, info = env.step(np.clip(act[:, 0], -1, 1))
        for k, v in (('obs', obs), ('act', act), ('rew', rew), ('cost', COST_SCALE * cost), ('obs_next', nxt)):
            buf[k].append(np.asarray(v, np.float32))
        air += info['air']
        obs = nxt
    assert done
    # (T, n, ...) -> 에피소드별로 이어 붙인다 (원본 replay 와 같은 순서)
    b = {k: np.swapaxes(np.stack(v), 0, 1).reshape(n * T, *np.stack(v).shape[2:]) for k, v in buf.items()}
    end = np.tile((np.arange(T) == T - 1).astype(np.float32), n)
    batch = Batch(obs=b['obs'], act=b['act'], rew=b['rew'], cost=b['cost'], terminate=end.copy(), trunc=end.copy(), obs_next=b['obs_next'])
    ep_cost = b['cost'].reshape(n, T).sum(1)
    return batch, dict(cost=float(ep_cost.mean()), reward=float(b['rew'].reshape(n, T).sum(1).mean()), air_m3=float((air * E.STEP_H).mean()))


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    torch.set_num_threads(1)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    splits, days = E.load()
    cap = E.caps(days, splits['dev'])
    env = E.DailyEnv(days, E.Dynamics(E.PREPARED / 'surrogate.pkl'), cap)
    cfg = make_config(a.seed, a.max_lambda)
    agent = Agent68(cfg, learn_repr=a.method == 'fcsrl')
    meta = dict(method=a.method, seed=a.seed, updates=a.updates, n_episodes=N_EPISODES, warmup=WARMUP_EPISODES, cost_scale=COST_SCALE,
                cost_limit=COST_LIMIT, config=cfg, cap=cap.tolist(), dev_days=len(splits['dev']), holdout_evaluated=False, test_evaluated=False,
                hashes={str(p): sha(p) for p in (Path(__file__), HERE / 'env68.py', E.PREPARED / 'surrogate.pkl', E.PREPARED / 'data.pkl',
                                                   FCSRL_ROOT / 'fcsrl/agent/ppo_lag_repr.py', HYPER)})
    (a.out / 'effective_config.json').write_text(json.dumps(meta, indent=1))
    # 원본 trainer: 무작위 행동 16 에피소드로 한 번 학습
    agent.train()
    batch, _ = collect(agent, env, splits['dev'], rng, WARMUP_EPISODES, random=True)
    agent.learn(agent.process_fn(batch, None), cfg['trainer']['batch_size'], cfg['trainer']['train_repeat'])
    log = []
    for u in range(1, a.updates + 1):
        tick = time.monotonic()
        batch, res = collect(agent, env, splits['dev'], rng, N_EPISODES)
        agent.update_lagrangian_multiplier(res['cost'])
        processed = agent.process_fn(batch, None)
        feasi = to_numpy(processed.feasi_score)
        losses = agent.learn(processed, cfg['trainer']['batch_size'], cfg['trainer']['train_repeat'])
        with torch.no_grad():
            on = agent.obs_normalizer(batch.obs[:2000])
            sigma = float(agent.actor(on, agent.fixed_encoder.zs(on))[1].mean())
        row = dict(update=u, **res, lagrg=float(agent.lagrg_updater.get_lagrg()), seconds=time.monotonic() - tick,
                   feasi_target_pos_pct=float(100 * (feasi > 1e-6).mean()), feasi_target_gt05_pct=float(100 * (feasi > 0.05).mean()), sigma=sigma,
                   **{k.replace('loss/', ''): float(np.mean(v)) for k, v in losses.items()})
        for k, v in row.items():
            assert np.isfinite(v), (k, v)
        log.append(row)
        if u % 10 == 0 or u == a.updates:
            (a.out / 'train_log.json').write_text(json.dumps(log, indent=0))
            (a.out / 'progress.json').write_text(json.dumps(dict(update=u, updates=a.updates, elapsed=time.monotonic() - t0)))
            print(f"u{u} viol%={res['cost']:.1f} air={res['air_m3']:.0f} lam={row['lagrg']:.3f} {row['seconds']:.1f}s", flush=True)
    agent.train(False)
    torch.save(agent.state(), a.out / 'model_final.pt')
    check = Agent68(cfg, learn_repr=a.method == 'fcsrl')
    check.load_state(torch.load(a.out / 'model_final.pt', weights_only=False))
    probe = env.reset(E.episodes(splits['dev'][:4]))
    assert np.array_equal(check.act_det(probe), agent.act_det(probe))
    out = dict(method=a.method, seed=a.seed, final_lagrg=float(agent.lagrg_updater.get_lagrg()))
    for split in ('dev', 'val'):
        res, states, airs, zones = E.rollout(env, splits[split], agent.act_det)
        out[split] = res
        np.savez_compressed(a.out / f'{split}_rollout.npz', states=states, air=airs, zones=zones)
        runs = []
        for k in EVAL_SAMPLE_SEEDS:
            torch.manual_seed(100_000 + 1_000 * a.seed + k)
            res_k, states, airs, zones = E.rollout(env, splits[split], agent.act_sample)
            runs.append(res_k)
            if k == 0:
                np.savez_compressed(a.out / f'{split}_stochastic_rollout.npz', states=states, air=airs, zones=zones)
        out[f'{split}_stochastic'] = {key: None if any(r[key] is None for r in runs) else float(np.mean([r[key] for r in runs])) for key in runs[0]}
        out[f'{split}_stochastic_runs'] = runs
    out['final_sigma'] = log[-1]['sigma']
    out['elapsed_seconds'] = time.monotonic() - t0
    assert all(sha(p) == v for p, v in meta['hashes'].items())
    (a.out / 'metrics.json').write_text(json.dumps(out, indent=1))
    print('DONE', json.dumps(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['fcsrl', 'ppolag'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=UPDATES)
    p.add_argument('--max-lambda', type=float, default=MAX_LAMBDA)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
