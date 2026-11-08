"""확장 2: FCSRL 의 feasibility 점수를 정책 학습의 보조 신호로 쓴다 (Lee 작성).

FCSRL 논문은 결론에서 "feasibility 점수를 정책 학습의 보조 신호로 사용해 모든
상황에서의 안전(state-wise safety)을 달성하는 것"을 향후 과제로 남겼다. 원본
구현에서 feasibility 점수는 표현(encoder)을 훈련하는 데만 쓰이고 정책 손실에는
들어가지 않는다. Lee/e02 에서 하루 평균 제약을 지켜도 순간 NH4 위반이 30% 에서
40% 였으므로, 바로 이 지점을 겨냥한 확장이다.

방법. 행동 a_t 를 한 뒤 실제로 도달한 다음 상태 s_t+1 의 feasibility 예측
F(z_t+1) 을 벌점으로 정책 이점에 더한다. 수집 중 이미 계산해 둔 값
(`buf['next_feasi']`, 느림보 encoder 와 feasibility head 로 계산)을 그대로 쓰므로
추가 계산이 없다.

    adjusted = base_advantage - weight * standardize(F(z_t+1)) / (1 + lambda)

왜 다음 상태인가. 현재 상태만의 함수 f(s_t) 를 벌점으로 쓰면 행동과 무관하므로
정책 경사의 기댓값이 0 이 되어 아무 효과가 없다. F(z_t+1) 은 선택한 행동이 만든
결과라서 행동에 따라 달라지고, "앞으로 위반할 확률이 높은 상태로 가는 행동"을
직접 억제한다. 이진 비용에서 F 는 명제 4.1 에 따라 앞으로 한 번이라도 위반할
확률로 읽을 수 있다.

표준화하는 이유. 원본 PPO 의 보상 이점은 표준화되어 크기가 약 1 이다. F 를 평균
0, 표준편차 1 로 맞추면 weight 가 곧 상대 가중치가 되어 사전에 값을 고정하기 쉽다.
평균을 빼는 것은 상태 의존 baseline 이라 경사의 기댓값을 바꾸지 않는다.
(1 + lambda) 로 나누는 것은 Yoo 의 CVaR 확장과 같은 규칙으로, 승수가 커질 때 이
항이 상대적으로 커지지 않게 한다.

이 항은 안전을 보장하지 않는다. 학습된 추정값에 기반한 실용적 벌점이며, 제약
만족의 증명이 아니다. 라그랑주 승수와 비용 제약은 그대로 둔다.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ppo_fcsrl import PPOFCSRLAgent


@dataclass(frozen=True)
class FeasibilityPenaltyConfig:
    weight: float = 0.1
    standardize: bool = True
    # relative=True: 벌점의 RMS 가 기본 이점의 RMS 의 정확히 weight 배가 되게 한다.
    # Lee/e05 에서 relative=False(표준화한 F 에 weight 를 곱하고 1 + lambda 로 나눔)는 lambda 가 커지면
    # 기본 이점이 함께 커져서 실제 크기가 명목 weight 의 1.5% 에서 18% 로 줄었다. 그 문제를 없앤다.
    relative: bool = False

    def __post_init__(self):
        if isinstance(self.weight, bool) or not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Feasibility penalty weight must be a finite nonnegative number')
        if type(self.standardize) is not bool or type(self.relative) is not bool:
            raise ValueError('standardize and relative must be booleans')


class FeasibilityPenaltyMixin:
    def __init__(self, *args, feasibility_penalty=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feasibility_penalty = feasibility_penalty or FeasibilityPenaltyConfig()

    def adjust_advantage(self, advantage, buf):
        c = self.feasibility_penalty
        if c.weight == 0:
            return advantage
        scores = np.asarray(buf['next_feasi'], dtype=np.float64)
        if scores.shape != advantage.shape or not np.isfinite(scores).all():
            raise ValueError('Missing or invalid next state feasibility scores')
        # 예측 feasibility 는 분포의 기댓값이라 구간 범위의 아래쪽에서 0 보다 약간 작을 수 있다.
        # 표준화해서 쓰므로 부호는 문제가 없다 (Lee/e06 첫 실행에서 음수 검사가 학습을 중단시켜 제거).
        if c.standardize:
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        else:
            scores = scores - scores.mean()
        base_rms = float(np.sqrt(np.mean(np.asarray(advantage, dtype=np.float64) ** 2)))
        if c.relative:
            penalty = c.weight * base_rms * scores
        else:
            penalty = c.weight * scores / (1 + self.lagrg)
        self.last_penalty_rms = float(np.sqrt(np.mean(penalty ** 2)))
        self.last_base_rms = base_rms
        return advantage - penalty.astype(advantage.dtype)

    def save(self, path):
        super().save(path)
        (Path(path) / 'feasibility_penalty.json').write_text(
            json.dumps(asdict(self.feasibility_penalty), indent=2))


class PPOFCSRLFeasibilityPenaltyAgent(FeasibilityPenaltyMixin, PPOFCSRLAgent):
    pass
