"""e72 학습: e70 학습 코드 (승수 이득 x 25/0.3, 적분 상한) 에서 승수 상한과 학습 한도 두 값만 바꾼다.

e71 에서 한도를 못 지킨 seed 대부분은 승수가 상한 20 에 붙어 있었다. 상한을 올려 승수가 더 누를 수 있게 하고,
학습 한도를 평가 한도 0.3 보다 낮게 (기본 0.2) 두어 seed 별 흔들림이 있어도 평가 한도 아래에 남게 한다.
PID 이득은 e70, e71 과 같은 값 (25 / 0.3 배) 을 쓴다. 평가, 판정 한도 0.3, 환경, 비용, 보상, 행동은 바꾸지 않는다.
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e70'))
import train70  # noqa: E402  (e68 학습 코드를 불러오고 이득과 적분 상한 패치를 건다)

T = train70.T
EVAL_LIMIT = 0.3
_make_config = T.make_config


def make_config(seed, max_lambda):
    cfg = _make_config(seed, max_lambda)
    cfg['Lagrangian']['eval_limit'] = EVAL_LIMIT
    return cfg


if __name__ == '__main__':
    assert train70.ORIGINAL_LIMIT == 25.0 and T.COST_LIMIT == EVAL_LIMIT and abs(train70.PID_SCALE - 25.0 / 0.3) < 1e-12
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['fcsrl', 'ppolag'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--max-lambda', type=float, required=True)
    p.add_argument('--train-limit', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert 0 < a.train_limit <= EVAL_LIMIT
    T.COST_LIMIT = a.train_limit  # constraint_threshold 와 기록용 cost_limit 에만 쓰인다. 이득 배율은 import 때 0.3 기준으로 이미 정해졌다
    T.make_config = make_config
    T.run(a)
