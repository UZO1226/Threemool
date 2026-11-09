"""안전 차단 (Park/e01 의 확장, Park 작성).

정책이 고른 행동을 환경에 넣기 직전에 확인하고, 위험하다고 판단하면 최대 폭기로 바꾼다.
evaluate.run(..., shield=...) 에 넘겨 쓴다. 모든 차단기는 shield(obs, action) -> (행동, 개입 여부) 다.

- FeasibilityShield: FCSRL 이 예측한 F(s, a) >= tau 이면 차단한다. 이 실험의 확장이다
- NH4Shield: 현재 방류 NH4 측정값(관측 0번) >= theta 이면 차단한다. 대조군
- RandomShield: 확률 p 로 무작위 차단한다. 대조군

대조군의 theta 와 p 는 FeasibilityShield 와 개입 비율이 같아지도록 학습 날짜에서 정한다.
"""
import numpy as np

from env import action_to_kla, kla_to_action

FALLBACK = np.array([kla_to_action(240.0)])   # 최대 폭기 KLa5 = 240, 행동으로는 +1


class FeasibilityShield:
    def __init__(self, agent, tau, fallback=FALLBACK):
        if not hasattr(agent, 'feasibility'):
            raise TypeError('FeasibilityShield 는 FCSRL agent 에만 쓸 수 있다')
        self.agent, self.tau, self.fallback = agent, float(tau), fallback

    def score(self, obs, action):
        o = self.agent.normalizer(np.atleast_2d(obs))
        return float(self.agent.feasibility(o, np.clip(action, -1, 1).reshape(1, -1))[0])

    def __call__(self, obs, action):
        if self.score(obs, action) >= self.tau:
            return self.fallback.copy(), True
        return action, False


class NH4Shield:
    def __init__(self, theta, fallback=FALLBACK):
        self.theta, self.fallback = float(theta), fallback

    def __call__(self, obs, action):
        if float(obs[0]) >= self.theta:
            return self.fallback.copy(), True
        return action, False


class RandomShield:
    def __init__(self, p, seed=0, fallback=FALLBACK):
        self.p, self.fallback = float(p), fallback
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, action):
        if self.rng.random() < self.p:
            return self.fallback.copy(), True
        return action, False


class PredictorShield:
    """예측기가 낸 위반 확률 >= threshold 이면 차단한다 (Park/e04).

    prob(obs) -> 0 ~ 1 사이 값 하나. 예측기는 실험 쪽에서 맞춰 넘긴다.
    """

    def __init__(self, prob, threshold, fallback=FALLBACK):
        self.prob, self.threshold, self.fallback = prob, float(threshold), fallback

    def __call__(self, obs, action):
        if self.prob(obs) >= self.threshold:
            return self.fallback.copy(), True
        return action, False


class GradedShield:
    """위험 확률에 비례해 폭기 하한을 올린다 (Park/e05).

    KLa5 = max(정책 KLa5, 240 x min(1, p / c)). 하한이 정책 행동보다 클 때를 개입으로 센다.
    """

    def __init__(self, prob, c):
        self.prob, self.c = prob, float(c)

    def __call__(self, obs, action):
        floor = 240.0 * min(1.0, self.prob(obs) / self.c)
        if floor > action_to_kla(float(np.asarray(action).reshape(-1)[0])):
            return np.array([kla_to_action(floor)]), True
        return action, False


def tau_to_minutes(tau, discount=0.9, minutes_per_step=5):
    """위반이 k 스텝 뒤에 확실히 일어나면 F = discount^k 다. tau 가 대략 몇 분 앞을 뜻하는지 환산한다."""
    return float(np.log(tau) / np.log(discount) * minutes_per_step)
