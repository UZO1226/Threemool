"""학습이 끝난 e63 정책 하나를 공통 v1.1 모델 (DO 대응 오류가 있는 팀 공통 모델) 에서 val park_e19 로 평가한다.

이득이 DO 수정 모델에만 기대는지 보는 교차 평가다. 차단도 공통 모델로 예측한다.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alloc as al  # noqa: E402
se, base = al.se, al.base
COMMON = se.PREPARED  # e50 준비물: 같은 데이터, prior, 정규화, 공통 대리 모델


def load(condition, method, seed, run_dir, prepared):
    c = base.common.config(66)
    agent = al.make_agent(method, seed, prepared, c) if condition == 'alloc' else base.common.make_agent(method, seed, prepared, c)
    agent.load(run_dir / 'model_final')
    return agent


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    agent = load(a.condition, a.method, a.seed, a.run_dir, a.prepared)
    fp = base.fingerprint(agent)
    splits, days = base.load_selected(COMMON / 'data.pkl')
    dyn = base.Dynamics(COMMON / 'surrogate.pkl')
    env = base.BatchEnv(days, dyn)
    result, rows, states, airs, zones = al.rollout(env, splits['val'], lambda o: agent.act_det(o), se.make_shield('park_e19', dyn))
    assert base.fingerprint(agent) == fp
    base.save(a.out / 'result.json', dict(condition=a.condition, method=a.method, seed=a.seed, model='common_v1_1', shield='park_e19',
                                          val=result, policy_fingerprint=fp, model_sha256=base.sha(COMMON / 'surrogate.pkl'), test_evaluated=False))
    np.savez_compressed(a.out / 'rollout.npz', states=states, air=airs, zones=zones)
    print('DONE', a.condition, a.method, a.seed, round(result['air_m3']), round(result['cost'], 3), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--condition', choices=['total', 'alloc'], required=True)
    p.add_argument('--method', choices=['ppo', 'fcsrl'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
