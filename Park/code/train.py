"""PPO-Lag 또는 FCSRL + PPO-Lag 를 학습하고 평가한다 (Park 작성).

수집 방식은 Lee/e01 과 같다. 환경 4개가 갱신마다 5일(1,440 스텝) 씩 모은다. 하루가 끝날
때마다 건기 0 ~ 6일 중 하나를 골라 그날 시작 상태로 reset 한다. 날 선택 난수는 seed 로 정한다.

    export BSM1_ARCHIVE=<BSM1.zip 경로>
    python train.py --method fcsrl --seed 0 --prepared <결과>/prepared --out <결과>/runs/fcsrl_s0
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm

from agent import FCSRL, METHODS, Config, ObsNormalizer, feasibility_targets, gae
from env import AerationEnv, STEPS_PER_DAY
from evaluate import evaluate

HERE = Path(__file__).resolve().parent
OBS_DIM, ACT_DIM = 11, 1
TRAIN_DAYS = 7


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def collect(agent, c, envs, day_states, rng):
    """갱신 한 번 분량을 모은다. 반환값은 (T, N) 배열 dict 와 진단값."""
    T, N = c.horizon, c.n_env
    schedule = rng.integers(0, TRAIN_DAYS, size=(T // STEPS_PER_DAY, N))
    keys = ('obs', 'obs_next', 'act', 'act_exec', 'logp', 'rew', 'cost', 'vr', 'vc',
            'nvr', 'nvc', 'end', 'terminated', 'next_feasi')
    b = {k: np.zeros((T, N), np.float32) for k in keys}
    for k in ('obs', 'obs_next'):
        b[k] = np.zeros((T, N, OBS_DIM), np.float32)
    for k in ('act', 'act_exec'):
        b[k] = np.zeros((T, N, ACT_DIM), np.float32)

    def start_day(i, block):
        day = int(schedule[block, i])
        envs[i].set_start(day_states[day], day)
        return envs[i].reset()[0]

    raw = np.stack([start_day(i, 0) for i in range(N)])
    for t in range(T):
        o = agent.normalizer(raw)
        with torch.no_grad():
            ot = agent.tensor(o)
            d = agent.dist(ot)
            act = d.sample()
            b['logp'][t] = d.log_prob(act).sum(-1).numpy()
            vr, vc = agent.values(ot)
            b['vr'][t], b['vc'][t] = vr.numpy(), vc.numpy()
        b['obs'][t], b['act'][t] = o, act.numpy()
        b['act_exec'][t] = np.clip(b['act'][t], -1, 1)
        nxt = np.empty_like(raw)
        for i, env in enumerate(envs):
            nxt[i], r, term, _, info = env.step(b['act_exec'][t, i])
            b['rew'][t, i], b['cost'][t, i] = r, info['cost']
            b['end'][t, i] = b['terminated'][t, i] = float(term)
        no = agent.normalizer(nxt)
        b['obs_next'][t] = no
        with torch.no_grad():
            vr, vc = agent.values(agent.tensor(no))
            alive = 1 - b['terminated'][t]
            b['nvr'][t], b['nvc'][t] = vr.numpy() * alive, vc.numpy() * alive
        if isinstance(agent, FCSRL):
            b['next_feasi'][t] = agent.feasibility_state(no)
        raw = nxt
        if (t + 1) % STEPS_PER_DAY == 0 and t + 1 < T:
            raw = np.stack([start_day(i, (t + 1) // STEPS_PER_DAY) for i in range(N)])

    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    b['feasi'] = feasibility_targets(b['cost'], b['end'], b['terminated'], b['next_feasi'],
                                     c.feasibility_discount)
    daily_cost = b['cost'].reshape(-1, STEPS_PER_DAY, N).sum(axis=1)
    diag = {'daily_cost_mean': float(daily_cost.mean()), 'daily_cost_max': float(daily_cost.max()),
            'train_step_violation_percent': float(b['cost'].mean() * 100),
            'feasi_target_mean': float(b['feasi'].mean()),
            'mean_kla5': float(((b['act_exec'] + 1) * 120).mean()),
            'raw_action_clip_percent': float((np.abs(b['act']) > 1).mean() * 100),
            'training_days': schedule.tolist()}
    flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in b.items()}
    return flat, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=10)
    p.add_argument('--feasibility-discount', type=float, default=Config.feasibility_discount,
                   help='F 의 할인율. 클수록 더 먼 미래의 위반까지 본다 (Park/e02)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    c = Config(updates=a.updates, feasibility_discount=a.feasibility_discount)
    normalizer = ObsNormalizer(np.load(a.prepared / 'train_observations.npy'))
    agent = METHODS[a.method](OBS_DIM, ACT_DIM, c, normalizer)
    initial = dict(np.load(a.prepared / 'initial.npz'))
    day_states = [dict(np.load(a.prepared / f'day_{d}.npz')) for d in range(TRAIN_DAYS)]
    envs = [AerationEnv(day_states[0], 'dry', start_day=0, days=1) for _ in range(c.n_env)]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:
        commit = None
    save_json(a.out / 'config.json', {'method': a.method, 'seed': a.seed, 'config': asdict(c),
                                      'code_commit': commit, 'normalizer': normalizer.state()})

    started = time.monotonic()
    policy = lambda obs: agent.act_det(obs)  # noqa: E731
    save_json(a.out / 'initial_evaluation.json', evaluate(initial, policy))

    rng = np.random.default_rng(10000 + a.seed)
    records = []
    for update in tqdm(range(1, c.updates + 1), desc=f'{a.method} seed {a.seed}'):
        tick = time.monotonic()
        buf, diag = collect(agent, c, envs, day_states, rng)
        info = agent.update(buf)
        agent.update_lagrangian(diag['daily_cost_mean'])
        records.append({'update': update, 'seconds': time.monotonic() - tick,
                        'lambda': agent.lagrangian, **diag, **info})
        save_json(a.out / 'training.json', records)

    agent.save(a.out / 'model_final')
    final = evaluate(initial, policy)
    save_json(a.out / 'metrics.json', {'method': a.method, 'seed': a.seed, 'final': final,
                                       'within_limit': {w: final[w]['week']['violating_steps_per_day'] <= c.cost_limit
                                                        for w in final},
                                       'elapsed_seconds': time.monotonic() - started})
    print('완료', a.out, flush=True)


if __name__ == '__main__':
    main()
