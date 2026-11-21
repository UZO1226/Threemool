"""e74 학습: e72 의 학습 방식 (e68 에이전트, PID 이득 x 25 / 평가 한도, 적분 상한, 승수 상한 100, 학습 한도 = 평가 한도 x 2/3,
갱신 500번) 을 새 비용 (env74: NH4 > 4 또는 예산 초과 스텝) 에 그대로 옮긴다. 새 비용에 맞춘 추가 조정은 하지 않는다.

e72 의 비율을 그대로 쓴다.
- PID 이득 배율: 원본 한도 25 / 평가 한도. e70 ~ e72 는 25 / 0.3, 여기서는 25 / 4.75.
- 학습 한도: e72 는 0.2 / 0.3 = 2/3 였다. 여기서는 4.75 x 2/3.
- 승수 상한 100, 적분 상한 묶음, 갱신 500번, 수집 규모, 망, 손실, 평가 (표본 행동 고정 seed 3개) 는 같다.
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'e70'))
import env74  # noqa: E402,F401  (env68 의 DailyEnv 와 rollout 을 바꿔 끼운다)
import train70  # noqa: E402  (e68 학습 코드를 불러오고 적분 상한 패치를 건다)

T = train70.T
EVAL_LIMIT = env74.EVAL_LIMIT
TRAIN_LIMIT = EVAL_LIMIT * 2.0 / 3.0
MAX_LAMBDA = 100.0
UPDATES = 500
PID_SCALE = train70.ORIGINAL_LIMIT / EVAL_LIMIT
_make_config = T.make_config   # train70.make_config (이득에 train70.PID_SCALE 을 곱한다)


def make_config(seed, max_lambda):
    cfg = _make_config(seed, max_lambda)
    cfg['Lagrangian']['eval_limit'] = EVAL_LIMIT
    cfg['Lagrangian']['cost'] = 'env74: 1[NH4 > 4 or 누적 > 380]'
    return cfg


if __name__ == '__main__':
    assert train70.ORIGINAL_LIMIT == 25.0 and T.E.DailyEnv is env74.DailyEnv74 and T.E.rollout is env74.rollout74
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['fcsrl', 'ppolag'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=UPDATES)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.max_lambda = MAX_LAMBDA
    train70.PID_SCALE = PID_SCALE      # train70.make_config 가 호출 때 이 값을 읽는다
    T.COST_LIMIT = TRAIN_LIMIT         # constraint_threshold 와 기록용 cost_limit
    T.make_config = make_config
    T.run(a)
