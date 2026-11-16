"""e67: 고정 배분 w 를 공기 0 정책 또는 총 공기만 정책에 덧씌워 검증 날짜에서 평가한다. e64 evaluate64.evaluate 를 그대로 쓴다."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e64'))
import evaluate64 as ev  # noqa: E402
base, se, al = ev.base, ev.se, ev.al
E65 = Path('/home/jio/fcsrl_src/results/e65/main/runs')


def fixed_act(base_act, w):
    """배분 축 a1 = w / W_MAX 이면 alloc.weight(a1) = w."""
    a1 = w / al.W_MAX
    assert abs(al.weight(np.array([a1]))[0] - w) < 1e-12

    def act(obs):
        a0 = np.asarray(base_act(obs), dtype=float).reshape(len(obs), -1)[:, 0]
        return np.column_stack([a0, np.full(len(obs), a1)])
    return act


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    splits, days = base.load_selected(ev.PREPARED / 'data.pkl')
    dyn = base.Dynamics(ev.MODELS[a.model])
    env = base.BatchEnv(days, dyn)
    agent, fp = None, None
    if a.base == 'zero':
        base_act = lambda o: np.full(len(o), -1.)  # noqa: E731
    else:
        root = ev.E63 if a.seed < 5 else E65
        agent, _ = ev.load_policy('total', 'ppo', a.seed, root)
        fp = base.fingerprint(agent)
        base_act = lambda o: agent.act_det(o)  # noqa: E731
    act = fixed_act(base_act, a.w)
    ev.make_act = lambda policy, ag, e: act  # evaluate64.evaluate 가 부르는 행동 함수를 고정 배분 행동으로 바꾼다
    result, rows, states, airs, zones = ev.evaluate(env, splits['val'], 'policy', agent, se.make_shield('park_e19', dyn), ev.caps(), 'daily')
    if agent is not None:
        assert base.fingerprint(agent) == fp
    base.save(a.out / 'result.json', dict(base=a.base, seed=a.seed, w=a.w, model=a.model, cap=True, mode='daily', shield='park_e19',
              policy_fingerprint=fp, val=result, elapsed_seconds=time.monotonic() - t0, test_evaluated=False))
    base.save(a.out / 'episodes.json', rows)
    print('DONE', a.base, a.seed, a.w, a.model, round(result['air_m3']), round(result['cost'], 3), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', choices=['zero', 'ppo_total'], required=True)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--w', type=float, required=True)
    p.add_argument('--model', choices=list(ev.MODELS), required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
