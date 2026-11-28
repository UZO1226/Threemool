"""e155: 같은 e50 정책 위에서 Park 차단과 Lee 위험 선택을 비교하고, 날짜 블록과 다른 대리 모델로 판정한다.

Park 코드 (차단 MinAirBrake 와 대표 설정 BRAKE, ABAC, 스텝 기록 rollout_log, 날짜 블록 block_ci) 는 Park 폴더에서
읽기 전용으로 불러온다. 위험 선택기는 Yoo/e135 의 risk.py (Lee e55, e56 의 4스텝 위험률 회귀) 를 쓴다.
정책은 e50 공통 v1.1 환경에서 차단 없이 학습한 PPO-Lag, FCSRL, FCSRL + Actor 보정 seed 0 ~ 4 이다. 새 학습은 없다.

    run.py prep  --out <결과>          # seed 별 위험 선택기 적합 (학습 188일, 정책 경로)
    run.py run   --out <결과> --jobs 18  # 모든 평가
    run.py report --out <결과>         # 표와 판정
test 41일은 불러오지 않는다 (e45 load_selected 는 train, val 만 꺼낸다).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / 'Park/code', ROOT / 'Park/experiments/e38', ROOT / 'Park/experiments/e49'):
    sys.path.insert(0, str(p))
import run_e38  # noqa: E402  (Park 대표 차단 설정 BRAKE)
from run_e49 import block_ci, series  # noqa: E402
from audit_brake import rollout_log  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
import darrow_env as DE  # noqa: E402
sys.path.insert(0, str(HERE.parent / 'e58'))
import severity as sev  # noqa: E402  (e59 와 같은 정책 불러오기 경로)
base = sev.base
sys.path.insert(0, str(HERE.parent / 'e135'))
from risk import FastRate, fit_models, slow_rates  # noqa: E402

RES = Path('/home/jio/fcsrl_src/results')
PREP = RES / 'e50/prepared'
RUNS = RES / 'e50/main/runs'
EVALUATORS = {
    'v11': PREP / 'surrogate.pkl',                                  # 공통 v1.1 (차단이 쓰는 모델과 같은 계열)
    'dofix': RES / 'e51/main/corrected_surrogate.pkl',              # DO 구간 대응을 고친 모델 (NH4, NO3 모델은 같음)
    'past148': RES / 'e139/main/prefix148/prepared/surrogate.pkl',  # 앞 148일만으로 적합한 모델 (10월 이후 날짜를 보지 않음)
}
METHODS = ('ppo', 'fcsrl', 'fcsrl_actor')
SEEDS = (0, 1, 2, 3, 4)
ABAC_SET = (3.2, 4000.0, 1600.0)            # Park/e54 다시 고른 ABAC (설정값, Kp, Ki), Park/e86 과 같음
OFFSETS = np.array([0, -.125, .125, -.25, .25])   # Yoo/e135, Lee/e63 후보
RISK_SEED = 155000
H2_SCALES = (900, 1100)
PARK_E86 = {  # Park/e86 results.md (공기량 반올림, 하루 비용 소수 셋째 자리)
    ('val', 'none'): (80673, 5.195), ('val', 'shield'): (80602, 4.625),
    ('train', 'none'): (69838, 2.577), ('train', 'shield'): (69414, 1.839),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale_action(action, milli):
    """e134 와 같은 공기량 배율 (milli / 1000)."""
    if milli == 1000:
        return action
    air = np.clip(DE.action_to_air(action) * (milli / 1000), 0, DE.AIR_MAX)
    return DE.air_to_action(air)


# ---- 작업자 준비 ----
G = {}


def setup():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)   # 시스템 libgomp 가 4 스레드로 잡혀 작업끼리 CPU 를 다툰다. 예측은 표본별이라 값은 같다
    base.DeviceConfig().select_device(-1)
    splits, days = base.load_selected(PREP / 'data.pkl')
    G['splits'] = splits
    G['dyn'] = {k: DE.Dynamics(p) for k, p in EVALUATORS.items()}
    G['env'] = {k: DE.BatchEnv(days, d) for k, d in G['dyn'].items()}
    G['config'] = base.common.config(66)


def load_policy(method, seed):
    agent = base.common.make_agent(method, seed, PREP, G['config'])
    path = RUNS / method / f'seed_{seed}' / 'model_final'
    agent.load(path)
    recorded = json.loads((path.parent / 'metrics.json').read_text())['val_final']
    return agent, recorded


def shield():
    return MinAirBrake(G['dyn']['v11'], 3.5, 1, 'others', **run_e38.BRAKE)


class RiskSelect:
    """Lee e63 원리 (Yoo/e135 LocalPolicy, 벌점 0): 정책 행동 주변 5 후보 중 예측 4스텝 위반률 합이 가장 작은 행동."""

    def __init__(self, agent, rate):
        self.agent, self.rate = agent, rate

    def __call__(self, obs):
        a = self.agent.act_det(obs)[:, 0]
        cand = np.clip(a[:, None] + OFFSETS[None, :], -1, 1)
        x = np.column_stack((np.repeat(obs, len(OFFSETS), axis=0), cand.ravel()))
        score = self.rate.predict(x).sum(axis=1).reshape(cand.shape)
        return cand[np.arange(len(a)), score.argmin(axis=1)]


# ---- 위험 선택기 적합 ----
def collect(agent, seed):
    """Yoo/e135 collect 와 같은 절차를 v1.1 환경과 학습 188일에서."""
    env = G['env']['v11']
    obs = env.reset(DE.all_episodes(G['splits']['train'], 5))
    rng = np.random.default_rng(RISK_SEED + seed)
    feats, labels = [], []
    for _ in range(env.T // 4):
        action = np.clip(agent.act_det(obs)[:, 0] + rng.choice(OFFSETS, size=len(obs)), -1, 1)
        feats.append(np.column_stack((obs, action)))
        counts = np.zeros((len(obs), 2))
        for step in range(4):
            if step:
                action = agent.act_det(obs)[:, 0]
            obs, _, _, _, info = env.step(action)
            counts += np.column_stack((info['v_nh4'], info['v_do']))
        labels.append(counts / 4)
    return np.concatenate(feats), np.concatenate(labels)


def prep_seed(args):
    out, seed = args
    parts = [collect(load_policy(m, seed)[0], seed) for m in METHODS]
    x, y = (np.concatenate(v) for v in zip(*parts))
    assert x.shape == (len(METHODS) * 188 * 5 * 23, 12), x.shape
    models = fit_models(x, y)
    test = x[::23]
    np.testing.assert_allclose(FastRate(models).predict(test), slow_rates(models, test), atol=1e-12, rtol=0)
    with (out / f'risk_s{seed}.pkl').open('wb') as f:
        pickle.dump(models, f)
    return dict(seed=seed, rows=len(x), nh4_rate=float(y[:, 0].mean()), do_rate=float(y[:, 1].mean()))


# ---- 평가 ----
def jobs_all():
    J = []
    ctrls = [('abac', 0, m) for m in ('none', 'shield')] + [(c, s, m) for c in METHODS for s in SEEDS for m in ('none', 'shield', 'risk')]
    for split, ev in (('train', 'v11'), ('val', 'v11'), ('val', 'dofix'), ('val', 'past148')):
        J += [dict(split=split, ev=ev, ctrl=c, seed=s, module=m, scale=1000) for c, s, m in ctrls]
    J += [dict(split='val', ev='v11', ctrl=c, seed=s, module='shield', scale=k) for c in ('ppo', 'fcsrl') for s in SEEDS for k in H2_SCALES]
    # 오래 걸리는 작업 (학습 188일, 차단) 을 먼저
    return sorted(J, key=lambda j: (j['split'] != 'train', j['module'] != 'shield'))


def name(j):
    who = 'abac' if j['ctrl'] == 'abac' else f"{j['ctrl']}_s{j['seed']}"
    return f"{j['split']}_{j['ev']}_{who}_{j['module']}_{j['scale']}"


def run_job(args):
    out, j, days = args
    n = name(j)
    env = G['env'][j['ev']]
    days = G['splits'][j['split']] if days is None else G['splits'][j['split']][:days]
    recorded = None
    if j['ctrl'] == 'abac':
        assert j['module'] != 'risk' and j['scale'] == 1000
        act = ABAC(env, *ABAC_SET)
    else:
        agent, recorded = load_policy(j['ctrl'], j['seed'])
        if j['module'] == 'risk':
            assert j['scale'] == 1000
            with (out / f"risk_s{j['seed']}.pkl").open('rb') as f:
                act = RiskSelect(agent, FastRate(pickle.load(f)))
        else:
            act = (lambda ag, k: (lambda obs: scale_action(ag.act_det(obs)[:, 0], k)))(agent, j['scale'])
    sh = shield() if j['module'] == 'shield' else None
    t0 = time.monotonic()
    r, L = rollout_log(env, days, act, shield=sh)
    r['seconds'] = time.monotonic() - t0
    if sh is not None:
        r.update(plan_steps=sh.plan_steps, conflict_steps=sh.conflict_steps, lowered_steps=sh.lowered_steps, steps=sh.steps)
    if recorded is not None and j['split'] == 'val' and j['ev'] == 'v11' and j['module'] == 'none' and j['scale'] == 1000 and days == G['splits']['val']:
        # e50 이 남긴 검증 지표를 재현해야 한다 (e59 와 같은 허용치: 공기 상대 1e-7, 비용 1e-9)
        r['reproduced_e50'] = bool(abs(r['air_m3'] - recorded['air_m3']) < 1e-7 * recorded['air_m3'] and abs(r['cost'] - recorded['cost']) < 1e-9)
    np.savez_compressed(out / f'log_{n}.npz', **L)
    (out / f'{n}.json').write_text(json.dumps(dict(job=j, result=r), indent=1))
    print('DONE', n, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    return n


def check_inputs():
    assert sha(ROOT / 'Park/code/darrow_env.py') == sha(HERE.parent / 'e30/darrow_env.py')
    assert Path(DE.__file__).resolve() in (ROOT / 'Park/code/darrow_env.py', HERE.parent / 'e30/darrow_env.py')
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
            'brake': {k: (str(v) if k == 'margins' else v) for k, v in run_e38.BRAKE.items()},
            'sha256': {k: sha(p) for k, p in list(EVALUATORS.items()) + [('data', PREP / 'data.pkl')]
                       + [(f.name, f) for f in (ROOT / 'Park/code/min_brake.py', ROOT / 'Park/code/audit_brake.py',
                                               ROOT / 'Park/code/baselines_darrow.py', HERE.parent / 'e135/risk.py', HERE / 'run.py')]}}


def cmd_prep(a):
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'manifest.json').write_text(json.dumps(check_inputs(), indent=1))
    with ProcessPoolExecutor(len(SEEDS), initializer=setup) as ex:
        rows = list(ex.map(prep_seed, [(a.out, s) for s in SEEDS]))
    (a.out / 'risk_fit.json').write_text(json.dumps(rows, indent=1))
    print(rows)


def cmd_run(a):
    J = [j for j in jobs_all() if not a.filter or a.filter in name(j)]
    J = [j for j in J if not (a.out / f'{name(j)}.json').exists()]
    print(len(J), 'jobs', flush=True)
    with ProcessPoolExecutor(a.jobs, initializer=setup) as ex:
        for n in ex.map(run_job, [(a.out, j, a.days) for j in J]):
            pass


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['prep', 'run'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=18)
    p.add_argument('--filter', default='')
    p.add_argument('--days', type=int, default=None, help='시험 실행용: 앞 날짜 수만')
    a = p.parse_args()
    {'prep': cmd_prep, 'run': cmd_run}[a.cmd](a)
