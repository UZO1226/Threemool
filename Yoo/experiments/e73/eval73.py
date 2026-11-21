"""e73 평가 한 건: 저장된 e72 정책 (또는 공기 0 정책) 을 예산 속도 차단과 함께 dev 또는 val 에서 돈다.

새 학습은 없다. 표본 행동은 e68 과 같은 고정 seed (100,000 + 1,000 x 정책 seed + k, k = 0, 1, 2) 로 뽑는다.
차단은 torch 난수를 쓰지 않으므로 같은 seed 에서 정책이 뽑는 난수 순서는 차단 없는 평가와 같다.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e68'))
sys.path.insert(0, str(HERE))
import env68 as E  # noqa: E402
import train68 as T  # noqa: E402
from shield73 import PaceShield, air0  # noqa: E402

E72_RUNS = Path('/home/jio/fcsrl_src/results/e72/main/runs')
SAMPLE_SEED = lambda seed, k: 100_000 + 1_000 * seed + k  # noqa: E731  (train68.run 과 같다)


def make_env():
    splits, days = E.load()
    return splits, E.DailyEnv(days, E.Dynamics(E.PREPARED / 'surrogate.pkl'), E.caps(days, splits['dev']))


def load_agent(method, seed, runs=E72_RUNS):
    run = runs / f'{method}_s{seed}'
    meta = json.loads((run / 'effective_config.json').read_text())
    assert meta['method'] == method and meta['seed'] == seed
    agent = T.Agent68(meta['config'], learn_repr=method == 'fcsrl')
    agent.load_state(torch.load(run / 'model_final.pt', weights_only=False))
    agent.train(False)
    return agent


def evaluate(env, dates, policy, margin, members='others', keep_log=False):
    """policy(obs) -> 행동. margin 이 None 이면 차단 없음. (지표, 차단 집계, 상태 배열, 차단 기록) 을 돌려준다."""
    shield = None
    if margin is not None:
        shield = PaceShield(env, margin, members=members)
        if not keep_log:
            shield.log = None
        policy = shield.wrap(policy)
    res, states, airs, zones = E.rollout(env, dates, policy)
    return res, (shield.counts() if shield else None), (states, airs), (shield.log if shield else None)


def main(a):
    torch.set_num_threads(1)
    splits, env = make_env()
    out = dict(policy=a.policy, seed=a.seed, split=a.split, margin=a.margin, runs=[])
    t0 = time.monotonic()
    if a.policy == 'air0':
        res, cnt, _, _ = evaluate(env, splits[a.split], air0, a.margin)
        out['runs'].append(dict(mode='det', **res, shield=cnt))
    else:
        agent = load_agent(a.policy, a.seed)
        for k in (0, 1, 2):
            torch.manual_seed(SAMPLE_SEED(a.seed, k))
            res, cnt, _, _ = evaluate(env, splits[a.split], agent.act_sample, a.margin)
            out['runs'].append(dict(mode=f'stoch{k}', **res, shield=cnt))
        res, cnt, _, _ = evaluate(env, splits[a.split], agent.act_det, a.margin)
        out['runs'].append(dict(mode='det', **res, shield=cnt))
    out['seconds'] = time.monotonic() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print('DONE', a.out, f"{out['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=['fcsrl', 'ppolag', 'air0'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', choices=['dev', 'val'], required=True)
    p.add_argument('--margin', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
