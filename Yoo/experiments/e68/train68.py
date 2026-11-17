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
2. 비용 척도: 환경 비용은 0/1 (하루 위반이 확정되는 스텝). 승수와 비용 가치에는 100 을 곱해 하루 위반율 % 단위로
   쓰고 한도 2.5 (%) 와 비교한다. 가능성 목표는 원본 정의대로 0/1 비용으로 계산한다. 원본 learn 은 비용 이점을
   보상 이점의 표준편차로 나누므로 100 은 단위 변환에 그치지 않고 승수 상한 1.0 에서의 최대 비용 가중치도 정한다
   (위반일 하나 = 승수 x 100 보상 단위, 하루 내내 공기 1,000 을 줄이면 보상 약 10.6).
3. 평가: 원본 test_agent 는 표본 행동으로 평가한다. 여기서는 결정적 행동 (평균을 [-1, 1] 로 자름) 을 주 평가로 쓰고,
   고정 seed 표본 행동 평가와 마지막 표준편차를 함께 남긴다.
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
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env68 as E  # noqa: E402

FCSRL_ROOT = Path('/home/jio/fcsrl_src/FCSRL')
sys.path.insert(0, str(FCSRL_ROOT))
from fcsrl.agent import PPOLagReprAgent  # noqa: E402
from fcsrl.data import Batch  # noqa: E402
from fcsrl.utils import MeanStdNormalizer, dict2attr, to_numpy  # noqa: E402

COST_SCALE = 100.0
COST_LIMIT = 2.5
N_EPISODES = 84
UPDATES = 250
WARMUP_EPISODES = 16
HYPER = FCSRL_ROOT / 'hyper_params/PPORepr_Lag.yaml'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_config(seed):
    cfg = yaml.safe_load(HYPER.read_text())
    cfg['network'].update(s_dim=E.OBS_DIM, a_dim=1, repr_type='FCSRL')
    cfg['agent']['action_range'] = [-1.0, 1.0]
    cfg['Lagrangian']['constraint_threshold'] = COST_LIMIT
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
        return super().train_encoder(batch) if self.learn_repr else {}

    @torch.no_grad()
    def act_det(self, raw_obs):
        o = self.obs_normalizer(raw_obs)
        mu, _ = self.actor(o, self.fixed_encoder.zs(o))
        return to_numpy(mu.clamp(-1, 1))[:, 0]

    @torch.no_grad()
    def act_sample(self, raw_obs):
        o = self.obs_normalizer(raw_obs)
        return to_numpy(self(Batch(obs=o))[0].act)[:, 0]

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
        nxt, rew, cost, done, info = env.step(act[:, 0])
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
    assert set(np.unique(ep_cost)) <= {0.0, COST_SCALE}
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
    cfg = make_config(a.seed)
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
        torch.manual_seed(100_000 + a.seed)
        out[f'{split}_stochastic'], *_ = E.rollout(env, splits[split], agent.act_sample)
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
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
