"""e18 준비: DO PI 운전으로 조건화한 시작 상태, 학습 날짜 snapshot, 관측 통계, 비용 한도, 기준 제어 평가

1. e16 정상상태에서 건기 14일을 DO PI(설정값 2)로 운전해 평가 시작 상태를 만든다
2. 그 상태에서 건기 0 ~ 6일을 DO PI 로 운전하며 하루 시작 snapshot(PI 적분 포함)과 관측을 저장한다
3. 학습 날짜에서 DO PI 의 하루 평균 비용을 재고, 이를 RL 비용 한도로 고정한다 (같은 수질에서 에너지 절감)
4. ABAC 격자를 학습 날짜에서 평가해 "평균 NH4 가 DO PI 이하" 인 조합 중 에너지가 가장 작은 것을 고른다
5. 고정 KLa 84, 240, DO PI, ABAC 를 14일 연속 평가(건기, 우기)한다
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
from tqdm import tqdm

from abac import ABAC, FF_GRID, GRID, FeedForward
from setpoint_env import SetpointEnv, SetpointPI, evaluate, setpoint_to_action, snapshot_with_pi, training_days

SP2 = setpoint_to_action(2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steady', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tanks', nargs='+', type=int, default=[5], help='DO PI 로 제어하는 조 (e18: 5, e19: 3 4 5)')
    a = p.parse_args()
    T = tuple(a.tanks)
    a.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings('error', category=ODEintWarning)
    t0 = time.monotonic()
    steady = dict(np.load(a.steady))

    env = SetpointEnv(steady, days=14., tanks=T)
    env.reset()
    for _ in tqdm(range(env.T), desc='DO PI 조건화 14일', mininterval=5):
        env.step(np.array([SP2]))
    initial = snapshot_with_pi(env.model, env.pi, 2.0, env.pis)
    np.savez_compressed(a.out / 'initial.npz', **initial)

    env = SetpointEnv(initial, days=7., tanks=T)
    obs, _ = env.reset()
    observations, states = [], []
    for t in tqdm(range(env.T), desc='학습 날짜 0 ~ 6일 DO PI', mininterval=5):
        if t % 288 == 0:
            s = snapshot_with_pi(env.model, env.pi, env.prev_sp, env.pis)
            np.savez_compressed(a.out / f'day_{t // 288}.npz', **s)
            states.append(s)
        observations.append(obs.copy())
        obs, _, _, _, _ = env.step(np.array([SP2]))
    np.save(a.out / 'train_observations.npy', np.asarray(observations))

    do_pi_train = training_days(states, lambda o, e: [SP2], tanks=T)
    obs_arr = np.asarray(observations)
    l0 = float((obs_arr[:, 5] * obs_arr[:, 6]).mean())
    frontier = {}
    for sp in tqdm((0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0), desc='고정 설정값 경계'):
        act = setpoint_to_action(sp)
        frontier[str(sp)] = training_days(states, lambda o, e, act=act: [act], tanks=T)['mean']
    (a.out / 'constant_setpoint_frontier.json').write_text(json.dumps(frontier, indent=1))
    limit = do_pi_train['mean']['cost']

    grid = []
    for g in tqdm(GRID, desc='ABAC 격자 (학습 날짜)'):
        r = training_days(states, ABAC(**g), tanks=T)
        grid.append({**g, **r['mean']})
    ok = [g for g in grid if g['flow_weighted_nh4'] <= do_pi_train['mean']['flow_weighted_nh4']]
    best = min(ok, key=lambda g: g['aeration_kwh_per_day']) if ok else None
    ff_grid = []
    for g in tqdm(FF_GRID, desc='유입 부하 비례 격자 (학습 날짜)'):
        r = training_days(states, FeedForward(l0=l0, **g), tanks=T)
        ff_grid.append({**g, **r['mean']})
    ok = [g for g in ff_grid if g['flow_weighted_nh4'] <= do_pi_train['mean']['flow_weighted_nh4']]
    ff_best = min(ok, key=lambda g: g['aeration_kwh_per_day']) if ok else None

    refs = {}
    policies = [('do_pi', lambda o, e: [SP2], None)]
    if T == (5,):  # 고정 KLa 기준은 5번 조만 제어하는 e18 구조에서만 의미가 있다
        policies = [('fixed84', lambda o, e: [84 / 120 - 1], None), ('fixed240', lambda o, e: [1.], None)] + policies
    if best:
        policies.append(('abac', ABAC(best['nh4_ref'], best['kp'], best['ti_hours']), None))
    if ff_best:
        policies.append(('feedforward', FeedForward(ff_best['a'], ff_best['b'], l0), None))
    for name, pol, _ in policies:
        if name.startswith('fixed'):
            refs[name] = evaluate_fixed_kla(initial, pol)
        else:
            refs[name] = evaluate(initial, pol, tanks=T)
        print(name, {w: {k: round(v, 3) for k, v in refs[name][w]['week'].items() if not isinstance(v, dict)}
                     for w in refs[name]}, flush=True)

    meta = {'created_at': datetime.now(timezone.utc).isoformat(),
            'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_hashes': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')},
            'steady_sha256': hashlib.sha256(a.steady.read_bytes()).hexdigest(),
            'archive_sha256': hashlib.sha256(Path(os.environ['BSM1_ARCHIVE']).read_bytes()).hexdigest(),
            'tanks': list(T), 'load_reference_l0': l0, 'cost_limit': limit, 'do_pi_training_days': do_pi_train,
            'abac_grid': grid, 'abac_selected': best, 'feedforward_grid': ff_grid, 'feedforward_selected': ff_best,
            'files': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.out.glob('*.np*'))},
            'seconds': time.monotonic() - t0}
    (a.out / 'metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
    (a.out / 'references.json').write_text(json.dumps(refs, indent=2, allow_nan=False))
    print('PREPARED cost_limit=%.3f abac=%s %.0fs' % (limit, best, time.monotonic() - t0), flush=True)


def evaluate_fixed_kla(initial, policy):
    """고정 KLa 기준은 e17 MeanEnv 로 평가한다 (PI 없이 KLa 를 직접 준다)"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'e17'))
    from bsm_mean import evaluate as eval_kla
    return eval_kla(initial, lambda o: policy(o, None))


if __name__ == '__main__':
    main()
