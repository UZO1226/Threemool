"""Kaggle 학습 날짜에서 고정한 운전 정책과 그 위의 PPO/FCSRL 보정."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / 'code/wwtp'
sys.path.insert(0, str(CODE))
from env import AerationEnv, AIR_MAX
from ppo_lag import PPOLagAgent, RunningNorm, mlp
from ppo_fcsrl import PPOFCSRLAgent


def observations(x):
    """실제 관측과 직전 공기량. 현재 공기량은 목표에만 쓰고 입력에는 넣지 않는다."""
    n = len(x['nh4']) - 1
    ph = x['phase'][:n]
    h = x['hour'][:n]
    prev = np.r_[0., x['air'][:n - 1]]
    return np.column_stack([x['nh4'][:n], x['do'][:n], ph == 0, ph == 1, ph == 2,
                           x['temp'][:n] / 10, x['q_in'][:n] / 1000,
                           np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                           np.minimum(x['since'][:n], 24) / 12, prev / AIR_MAX]).astype(np.float32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data, model, raw, out):
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    expected = {'raw': 'b57bc042a394d84da9fc9620aae5b90f029885dbafa14d3ce6ed437a0b005a2e',
                'data': '3a01d5995e49d3b4f3fd42240a4579d1be991629ef70243435422f797d789315',
                'model': 'd25dbe3ea1a8e33b17f0a2d0f11a5983fe6c4a6d81a1d0db851b530386658117'}
    paths = {'raw': raw, 'data': data, 'model': model}
    for key, path in paths.items():
        if sha(path) != expected[key]:
            raise ValueError(f'Existing Kaggle input hash changed: {key}')
    torch.set_num_threads(1)
    torch.manual_seed(2300)
    env = AerationEnv(data, model, split='train')
    days = sorted(env.days)
    assert len(days) == 330
    fit_days, check_days = days[:264], days[264:]

    def arrays(ds):
        obs = np.concatenate([observations(env.day_data[d]) for d in ds])
        target = np.concatenate([np.clip(env.day_data[d]['air'][:-1] / AIR_MAX * 2 - 1, -1, 1) for d in ds])
        return obs, target

    x, y = arrays(fit_days)
    xv, yv = arrays(check_days)
    norm = RunningNorm(11)
    norm.update(x)
    # 이진 단계 및 하루 주기 입력의 0 분산을 작은 분모로 확대하지 않는다.
    norm.m2 = np.maximum(norm.m2, norm.n * .05 ** 2)
    train_mask, check_mask = x[:, 4] == 1, xv[:, 4] == 1
    tx = torch.tensor(norm(x[train_mask]), dtype=torch.float32)
    ty = torch.tensor(y[train_mask, None], dtype=torch.float32)
    vx = torch.tensor(norm(xv[check_mask]), dtype=torch.float32)
    net = mlp(11, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    log = []
    for epoch in range(40):
        for ids in torch.randperm(len(tx)).split(1024):
            loss = ((net(tx[ids]) - ty[ids]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(vx)[:, 0].clamp(-1, 1).numpy()
        log.append({'epoch': epoch + 1, 'check_air_mae': float(np.mean(abs(pred - yv[check_mask])) * AIR_MAX / 2)})
    # 고정 40 epoch의 마지막 모델을 사용한다. 확인 오차로 후보를 고르지 않는다.
    out.mkdir(parents=True)
    torch.save(net.state_dict(), out / 'prior.pt')
    (out / 'normalizer.json').write_text(json.dumps(norm.state()))
    report = {'source': 'Kaggle jueyetao/wastewater-treatment-dataset, Avedore',
              'input_sha256': {str(path.resolve()): sha(path) for path in paths.values()},
              'fit_days': fit_days, 'check_days': check_days, 'seed': 2300, 'epochs': 40,
              'train_aerobic_samples': int(train_mask.sum()), 'check_aerobic_samples': int(check_mask.sum()),
              'check_air_mae': log[-1]['check_air_mae'],
              'previous_air_check_mae': float(np.mean(abs(xv[check_mask, -1] * AIR_MAX - (yv[check_mask] + 1) * AIR_MAX / 2))),
              'log': log}
    (out / 'preparation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ('train_aerobic_samples', 'check_aerobic_samples', 'check_air_mae', 'previous_air_check_mae')}, indent=2))


class PriorMixin:
    def prior_action(self, normalized_obs):
        with torch.no_grad():
            return self.prior(normalized_obs).clamp(-1, 1)

    def dist(self, o):
        mean = self.prior_action(o) + self.pi(self.policy_input(o))
        return torch.distributions.Normal(mean, self.log_std.exp())

    @torch.no_grad()
    def act_det(self, obs):
        o = self._t(self.obs_normalizer(np.atleast_2d(obs)))
        return self.dist(o).mean.clamp(-1, 1).cpu().numpy()

    def save(self, path):
        super().save(path)
        torch.save(self.prior.state_dict(), Path(path) / 'prior.pt')

    def load(self, path):
        super().load(path)
        self.prior.load_state_dict(torch.load(Path(path) / 'prior.pt', map_location=self.dev, weights_only=True))


class PriorPPO(PriorMixin, PPOLagAgent):
    pass


class PriorFCSRL(PriorMixin, PPOFCSRLAgent):
    pass


def make_agent(method, seed, prepared, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = PriorFCSRL if method.startswith('fcsrl') else PriorPPO
    agent = cls(11, 1, config=config)
    agent.prior = mlp(11, 1).to(agent.dev)
    agent.prior.load_state_dict(torch.load(prepared / 'prior.pt', map_location=agent.dev, weights_only=True))
    agent.prior.requires_grad_(False).eval()
    agent.obs_normalizer.load(json.loads((prepared / 'normalizer.json').read_text()))
    with torch.no_grad():
        agent.pi[-1].weight.zero_()
        agent.pi[-1].bias.zero_()
    return agent


def behavior_penalty(actions, prior, aerobic):
    """실제로 전달한 정규화 공기량과 기록 정책의 차이. 비호기 단계의 실행 공기량은 항상 0이다."""
    return np.asarray(aerobic, dtype=float) * (np.clip(actions, -1, 1) - np.clip(prior, -1, 1)) ** 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('data', 'model', 'raw', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.data, a.model, a.raw, a.out)
