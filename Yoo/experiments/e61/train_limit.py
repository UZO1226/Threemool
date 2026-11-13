"""e61: e60 과 같은 학습 중 차단 학습에서 학습 비용 한도만 12 에서 낮춘다.

e60/train_shielded.py 를 수정하지 않고 그대로 실행한다. 바꾸는 것은 e30 common.config 가 돌려주는 설정의
cost_limit 하나다. 이 값은 PID 승수 갱신 (learner.pid.update(..., c.cost_limit)) 에만 쓰인다.
평가 지표의 한도 12 와 Park 비교 기준은 바뀌지 않는다.
"""
import dataclasses
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e60'))
import train_shielded as t60  # noqa: E402

common = t60.base.common
ORIGINAL_CONFIG = common.config


def install(limit):
    """common.config 를 한도만 바꾼 설정을 돌려주도록 바꾼다. 다른 모든 값은 원래 설정과 같아야 한다."""
    limit = float(limit)
    assert 0 < limit < 12

    def config(updates=66):
        c = ORIGINAL_CONFIG(updates)
        out = dataclasses.replace(c, cost_limit=limit)
        assert {k: v for k, v in out.to_dict().items() if k != 'cost_limit'} == {k: v for k, v in c.to_dict().items() if k != 'cost_limit'}
        return out
    common.config = config
    return config


if __name__ == '__main__':
    # --train-cost-limit 를 꺼낸 뒤 나머지 인자는 e60 실행기에 그대로 넘긴다.
    args = sys.argv[1:]
    i = args.index('--train-cost-limit')
    limit = float(args[i + 1])
    del args[i:i + 2]
    install(limit)
    assert t60.base.common.config(66).cost_limit == limit
    # e60 train_shielded.py 의 __main__ 과 같은 인자. resume 은 쓰지 않지만 t60.run 이 읽는다.
    p = t60.argparse.ArgumentParser()
    p.add_argument('--method', choices=['ppo', 'fcsrl'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--train-shield', default=t60.TRAIN_SHIELD, choices=[k for k in t60.se.SHIELDS if k != 'none'])
    p.add_argument('--prepared', type=Path, default=t60.se.PREPARED)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(args)
    assert not a.resume, 'e61 은 재개를 쓰지 않는다 (resume metadata 에 한도가 없다)'
    t60.run(a)
