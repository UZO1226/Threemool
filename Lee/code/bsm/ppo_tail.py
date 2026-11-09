"""완결된 하루 비용의 표본 CVaR 경사를 PPO에 결합하는 확장 후보.

Tamar, Glassner, Mannor (AAAI 2015), https://arxiv.org/abs/1404.3862 의
likelihood-ratio 경사에서 VaR를 뺀 초과 비용을 사용한다. 비용의 upper tail이다.
TRC나 WCSAC 전체 알고리즘을 재현하지 않으며 정규분포 근사도 사용하지 않는다.
PPO clipping과 유한 표본 분위수 때문에 실용적인 local surrogate이며 안전 보장은 없다.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ppo_lag import PPOLagAgent
from ppo_fcsrl import PPOFCSRLAgent


@dataclass(frozen=True)
class TailConfig:
    tail_fraction: float = 0.2
    weight: float = 0.03
    cost_scale: float = 15.0

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError('Tail parameters must be finite numbers')
        if not 0 < self.tail_fraction <= 1 or self.weight < 0 or self.cost_scale <= 0:
            raise ValueError('Invalid tail fraction, weight, or cost scale')


def empirical_cvar(costs, fraction=0.2):
    """upper-tail CVaR. 경계에 걸친 표본의 질량을 분수로 반영한다."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not len(costs) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Expected a nonempty vector of finite nonnegative episode costs')
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Tail fraction must be in (0, 1]')
    ordered = np.sort(costs)
    # empirical inverse CDF의 왼쪽 끝. 정수 tail 크기일 때도 VaR baseline을 유지한다.
    index = max(0, int(np.ceil((1 - fraction) * len(costs) - 1e-12)) - 1)
    var = float(ordered[index])
    return var, float(var + np.maximum(costs - var, 0).mean() / fraction)


def episode_tail_scores(cost, end, config):
    """각 trajectory의 (C - VaR)+ / (tail_fraction * cost_scale)를 모든 행동에 배정한다.

    비용 GAE에 tail 가중치를 곱하는 방법과 다르다. 같은 날의 전체 likelihood score를
    사용해야 하므로 rollout 시작과 끝은 모두 reset 경계여야 하며 하루 길이는 같아야 한다.
    """
    cost, end = np.asarray(cost), np.asarray(end)
    if cost.ndim != 2 or end.shape != cost.shape or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError('Expected finite nonnegative costs and matching (T, N) episode boundaries')
    if not np.isin(end, (0, 1)).all():
        raise ValueError('Episode boundaries must be binary')
    trajectories = []
    for env in range(cost.shape[1]):
        start = 0
        for last in np.flatnonzero(end[:, env]):
            trajectories.append((env, start, last + 1, float(cost[start:last + 1, env].sum())))
            start = last + 1
        if start != len(cost):
            raise ValueError('Incomplete episodes cannot be used for daily CVaR')
    if not trajectories or len({stop - start for _, start, stop, _ in trajectories}) != 1:
        raise ValueError('Daily trajectories must have equal positive lengths')
    returns = np.array([row[3] for row in trajectories])
    var, cvar = empirical_cvar(returns, config.tail_fraction)
    scores = np.maximum(returns - var, 0) / (config.tail_fraction * config.cost_scale)
    per_step = np.zeros_like(cost, dtype=np.float32)
    for (env, start, stop, _), score in zip(trajectories, scores):
        per_step[start:stop, env] = score
    return per_step, {'train_cost_var': var, 'train_cost_cvar': cvar,
        'tail_episode_count': len(returns), 'tail_effective_mass': len(returns) * config.tail_fraction,
        'tail_positive_episodes': int((scores > 0).sum()), 'tail_score_rms': float(np.sqrt(np.mean(scores ** 2)))}


def risk_metrics(rows, fraction=0.2):
    costs = np.array([row['cost'] for row in rows], dtype=float)
    var, cvar = empirical_cvar(costs, fraction)
    return {'cost_var': var, 'cost_cvar': cvar, 'tail_fraction': fraction}


class TailMixin:
    def __init__(self, *args, tail_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_config = tail_config or TailConfig()

    def adjust_advantage(self, advantage, buf):
        t = self.tail_config
        if t.weight == 0:
            return advantage
        scores = np.asarray(buf['tail_score'])
        if scores.shape != advantage.shape or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('Missing or invalid episode tail scores')
        # 기존 reward normalization, 비용 advantage 및 PID는 그대로 둔다.
        # 표본 VaR를 이미 뺐으므로 tail 내부에서 다른 baseline을 다시 빼지 않는다.
        return advantage - t.weight * scores / (1 + self.lagrg)

    def save(self, path):
        super().save(path)
        (Path(path) / 'tail_config.json').write_text(json.dumps(asdict(self.tail_config), indent=2))


class PPOTailAgent(TailMixin, PPOLagAgent):
    pass


class PPOFCSRLTailAgent(TailMixin, PPOFCSRLAgent):
    pass
