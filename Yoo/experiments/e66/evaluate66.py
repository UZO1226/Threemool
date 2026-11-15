"""e66: test 41일 최종 평가 한 건. e64 evaluate64.evaluate 를 그대로 쓰고 날짜만 test 로 바꾼다.

--split val 은 사전 검사 전용이다 (e64 결과 재현). 본 평가는 run.py 가 잠금 파일을 만든 뒤에만 --split test 로 부른다.
"""
import argparse
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e64'))
import evaluate64 as ev  # noqa: E402
base, se = ev.base, ev.se
from darrow_env import load_days  # noqa: E402  (e30 darrow_env, 모든 분할의 날짜 자료)

LOCK = Path('/home/jio/fcsrl_src/results/e66/TEST_USED.json')
import numpy as np  # noqa: E402


def main(a):
    if a.split == 'test':
        assert LOCK.exists(), 'test 평가는 run.py 로만 실행한다'
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    splits, days = load_days(ev.PREPARED / 'data.pkl')
    dates = list(splits[a.split])
    assert len(dates) == {'val': 40, 'test': 41}[a.split]
    dyn = base.Dynamics(ev.MODELS[a.model])
    env = base.BatchEnv(days, dyn)
    agent, path, fp = None, None, None
    if a.policy == 'policy':
        agent, path = ev.load_policy(a.condition, a.method, a.seed, a.runs_root)
        fp = base.fingerprint(agent)
    shield = None if a.shield == 'none' else se.make_shield(a.shield, dyn)
    cap = ev.caps() if a.cap else None
    result, rows, states, airs, zones = ev.evaluate(env, dates, a.policy, agent, shield, cap, a.mode)
    if agent is not None:
        assert base.fingerprint(agent) == fp
    base.save(a.out / 'result.json', dict(split=a.split, policy=a.policy, condition=a.condition, method=a.method, seed=a.seed, model=a.model,
              cap=bool(a.cap), mode=a.mode, shield=a.shield, days=len(dates), first_day=dates[0], last_day=dates[-1],
              checkpoint=None if path is None else str(path), policy_fingerprint=fp, val=result if a.split == 'val' else None,
              test=result if a.split == 'test' else None, model_sha256=base.sha(ev.MODELS[a.model]), elapsed_seconds=time.monotonic() - t0))
    base.save(a.out / 'episodes.json', rows)
    np.savez_compressed(a.out / 'rollout.npz', states=states, air=airs, zones=zones)
    print('DONE', a.split, a.policy, a.condition, a.method, a.seed, a.model, a.mode, round(result['air_m3']), round(result['cost'], 3), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--policy', choices=['policy', 'record', 'zero'], default='policy')
    p.add_argument('--condition', choices=['total', 'alloc', 'none'], default='none')
    p.add_argument('--method', choices=['ppo', 'fcsrl', 'none'], default='none')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--model', choices=list(ev.MODELS), required=True)
    p.add_argument('--cap', type=int, choices=[0, 1], required=True)
    p.add_argument('--mode', choices=['daily', 'continuous'], required=True)
    p.add_argument('--shield', choices=['none', 'park_e10', 'park_e19'], default='park_e19')
    p.add_argument('--runs-root', type=Path, default=ev.E63)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
