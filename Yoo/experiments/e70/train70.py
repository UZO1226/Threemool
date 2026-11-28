"""e70 학습: e68 의 train68.py 를 그대로 쓰고 PID 승수 이득 (KP, KI, KD) 만 원본 한도와 우리 한도의 비로 키운다.

원본 PPORepr_Lag.yaml 의 이득은 SafetyGym 비용 한도 25 에 맞춘 값이다. e68 한도는 0.3 이라 같은 상대 오차에서 오차가 약 83배 작고,
승수도 그만큼 느리게 움직인다 (e69 FCSRL 갱신당 약 0.0007). 이득에 25 / 0.3 을 곱하면 한도 대비 상대 오차에 원본과 같은 속도로 반응한다.
이득만 키우면 학습 초반 (비용이 한도의 수십 배) 에 적분항이 쌓여 승수가 상한 20 에 붙은 채 내려오지 않았다 (시험 학습).
그래서 원본 PIDLagrangianUpdater 의 적분 하한 묶음에 상한 묶음 (적분항 <= 승수 상한) 을 더한다. 나머지 계산은 원본과 같다.
비용 크기, 가치망, 보상, 행동, 평가는 바꾸지 않는다. 인자는 train68.py 와 같다.
"""
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e68'))
import train68 as T  # noqa: E402
from fcsrl.utils import PIDLagrangianUpdater  # noqa: E402

ORIGINAL_LIMIT = float(yaml.safe_load(T.HYPER.read_text())['Lagrangian']['constraint_threshold'])
PID_SCALE = ORIGINAL_LIMIT / T.COST_LIMIT
_make_config = T.make_config


def make_config(seed, max_lambda):
    cfg = _make_config(seed, max_lambda)
    for k in ('KP', 'KI', 'KD'):
        cfg['Lagrangian'][k] = float(cfg['Lagrangian'][k]) * PID_SCALE
    cfg['Lagrangian']['pid_scale'] = PID_SCALE
    return cfg


def pid_update(self, curr_cost, thres):
    """원본 solver.py update 와 같고, 적분 상한 묶음 (KI * 적분 <= max_lag) 한 줄만 더했다."""
    error_new = np.mean(curr_cost - thres)
    error_diff = max(0, error_new - self.error_old)
    self.error_integral = min(self.max_lag / self.KI, max(self.min_integral, self.error_integral + error_new))
    self.error_old = error_new
    self.lag = max(0, self.KP * error_new + self.KI * self.error_integral + self.KD * error_diff)
    self.lag = min(self.lag, self.max_lag)


T.make_config = make_config
PIDLagrangianUpdater.update = pid_update

if __name__ == '__main__':
    assert ORIGINAL_LIMIT == 25.0 and T.COST_LIMIT == 0.3, (ORIGINAL_LIMIT, T.COST_LIMIT)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['fcsrl', 'ppolag'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=T.UPDATES)
    p.add_argument('--max-lambda', type=float, default=T.MAX_LAMBDA)
    p.add_argument('--out', type=Path, required=True)
    T.run(p.parse_args())
