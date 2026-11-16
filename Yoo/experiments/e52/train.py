"""e52: 대리 모델 구성원 불일치로 NH4 제약을 비관적으로 세는 학습 (Codex e44 환경 그대로)

e44 의 수집, 갱신, 에이전트(관측 16, padded), e35 DO 구간 수정 환경, NH4 / DO 분리 한도 6 / 6 을 그대로 쓴다.
바꾼 것은 학습 중 NH4 비용 하나다.

    학습 NH4 비용 = 1[ NH4_next(선택 구성원) + kappa x std_m(NH4_next 의 구성원 5개 한 스텝 예측) > 4 ]

전이(실제로 다음 상태가 되는 값)는 바꾸지 않는다. 평가는 비관 없이 원래 비용으로 한다. kappa = 0 이면 e44 와 같다.
kappa 는 학습 날짜만으로 정했다: 기록 운전 재생의 비관 NH4 비용이 실제 기록 NH4 위반(5.62)과 같아지는 값 (README).
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
E44 = HERE.parent / 'e44'
sys.path.insert(0, str(E44))
from command import config, make_agent, LIMITS, save, CommandEnv  # noqa: E402
from fcsrl.utils import DeviceConfig  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent  # noqa: E402
from darrow_env import Dynamics, load_days, action_to_air, NH4_MAX  # noqa: E402

_spec = importlib.util.spec_from_file_location('e44_train', E44 / 'train.py')
e44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e44)


class UncertainEnv(CommandEnv):
    """학습 전용. 정보(info) 의 v_nh4 와 비용만 비관적으로 바꾸고 상태 전이는 그대로 둔다"""

    def __init__(self, days, dynamics, kappa):
        super().__init__(days, dynamics, 'padded')
        self.kappa = kappa
        self.sd_log = []

    def step(self, action):
        t = self.t
        air = action_to_air(np.asarray(action).reshape(-1))
        zone = air[:, None] * self.x['frac'][:, t]
        x = np.column_stack([self.state, self.x['exo'][:, t], zone / 1000.0, self.prev_zone / 1000.0])
        sd = np.stack([self.dyn.models[m][0].predict(x) for m in range(self.dyn.n_members)]).std(0) if self.kappa else 0.
        obs, reward, cost, done, info = super().step(action)
        if self.kappa:
            pess = ((self.state[:, 0] + self.kappa * sd) > NH4_MAX).astype(float)
            assert np.all(pess >= info['v_nh4'])
            info = dict(info, v_nh4=pess, v_nh4_plain=info['v_nh4'])
            cost = pess + info['v_do']
            self.sd_log.append(float(np.mean(sd)))
        return obs, reward, cost, done, info


def train(a):
    a.out.mkdir(parents=True, exist_ok=False)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    c = config(a.updates)
    agent = make_agent(a.method, a.seed, a.prepared, c)
    splits, days = load_days(a.prepared / 'data.pkl')
    dyn = Dynamics(a.corrected)
    env = UncertainEnv(days, dyn, a.kappa)
    plain = CommandEnv(days, dyn, 'padded')
    save(a.out / 'effective_config.json', {'method': a.method, 'kappa': a.kappa, 'seed': a.seed, 'corrected': str(a.corrected),
         'config': c.to_dict(), 'cost_limits': LIMITS.tolist(), 'prepared': str(a.prepared), 'selection': 'last',
         'evaluation': 'plain cost, train and val'})
    norm = agent.obs_normalizer.state()
    prior = {k: v.clone() for k, v in agent.prior.state_dict().items()}
    start, records = time.monotonic(), []
    rng = np.random.default_rng(a.seed)
    for u in tqdm(range(1, c.n_updates + 1), desc=f'e52 {a.method} k{a.kappa:g} s{a.seed}', mininterval=10):
        env.sd_log = []
        buf, diag = e44.collect(agent, c, env, splits['train'], rng)
        info = agent.update(buf)
        assert all(np.isfinite(v) for v in info.values())
        modules = [agent.pi, agent.vr, agent.vc]
        if isinstance(agent, PPOFCSRLAgent):
            modules += [agent.encoder, agent.fixed_encoder, agent.feasi_head, agent.proj_layer, agent.post_proj]
        assert all(torch.isfinite(p).all() for m in modules for p in m.parameters())
        agent.update_constraints([diag['nh4_cost'], diag['do_cost']])
        records.append({'update': u, 'lambda_nh4': float(agent.multipliers[0]), 'lambda_do': float(agent.multipliers[1]),
                        'mean_member_sd': float(np.mean(env.sd_log)) if env.sd_log else 0., **diag, **info})
        elapsed = time.monotonic() - start
        save(a.out / 'train_log.json', records)
        save(a.out / 'progress.json', {'phase': 'training', 'update': u, 'updates': c.n_updates, 'steps': u * 6080,
             'total_steps': c.actual_steps, 'elapsed_seconds': elapsed,
             'estimated_remaining_seconds': elapsed / u * (c.n_updates - u)})
    assert norm == agent.obs_normalizer.state()
    assert all(torch.equal(v, prior[k]) for k, v in agent.prior.state_dict().items())
    agent.save(a.out / 'model_final')
    results = {}
    for split in ('val', 'train'):
        save(a.out / 'progress.json', {'phase': 'evaluating_' + split, 'steps': c.actual_steps})
        summary, rows, _, _ = e44.evaluate(plain, splits[split], lambda x: agent.act_det(x)[:, 0])
        save(a.out / f'{split}_episodes.json', rows)
        results[split] = summary
    save(a.out / 'metrics.json', {'method': a.method, 'kappa': a.kappa, 'seed': a.seed, 'val_final': results['val'],
         'train_final': results['train'], 'steps': c.actual_steps, 'normalizer_and_prior_frozen': True,
         'elapsed_seconds': time.monotonic() - start})
    save(a.out / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps})
    print('DONE', results['val'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=('ppo', 'fcsrl'), required=True)
    p.add_argument('--kappa', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    train(p.parse_args())
