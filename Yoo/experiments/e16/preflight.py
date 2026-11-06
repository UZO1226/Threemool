"""BSM1 안정화, 고정 제어 대조, 환경 경계와 기존 agent 연결을 검사한다."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
from tqdm import tqdm
from bsm2_python import BSM1OL

from bsm_env import BSMAerationEnv, BASE_KLAS, PACKAGE, make_plant, snapshot, restore

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Yoo/code/wwtp'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def condition(model, steps, label):
    for i in tqdm(range(steps), desc=label, mininterval=5):
        model.step(i, BASE_KLAS.copy())
    return snapshot(model)


def steady(out):
    # upstream tests/bsm1_ol_test.py의 상수 유입과 MATLAB 기준값.
    # 출처와 BSD 라이선스는 README 및 BSM_LICENSE에 기록한다.
    row = np.array([0., 30, 69.5, 51.2, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95,
                    10.59, 7, 211.2675, 18446, 15, 0, 0, 0, 0, 0])
    last = row.copy()
    last[0] = 200.1
    m = BSM1OL(data_in=np.stack([row, last]), timestep=15 / 1440, endtime=200.)
    state = condition(m, len(m.simtime), '200 day stabilization')
    expected = np.array([30., .889492799653682, 4.39182747787874, .188440413683379,
        9.78152406404732, .572507856962265, 1.72830016782928, .490943515687561,
        10.4152201204309, 1.73333146817512, .688280004678034, .0134804685779854,
        4.12557938198182, 12.4969499853007, 18061., 15., 0, 0, 0, 0, 0])
    np.testing.assert_allclose(m.ys_eff, expected, rtol=1e-5, atol=1e-5)
    np.savez_compressed(out / 'steady_state.npz', **state)
    return state, {'matlab_reference_pass': True, 'max_abs_error': float(abs(m.ys_eff - expected).max()),
                   'effluent_nh4': float(m.ys_eff[9]), 'reactor5_do': float(m.y_out5[7])}


def rollout(state, weather, kla, minutes=1., days=7.):
    env = BSMAerationEnv(state, weather, days=days, substep_minutes=minutes)
    env.reset()
    energy = mixing = reward = cost = 0.
    nh4, do, flow = [], [], []
    start = time.perf_counter()
    for _ in tqdm(range(env.T), desc=f'{weather} KLa={kla:g} dt={minutes:g}', mininterval=5):
        _, r, term, trunc, info = env.step(np.array([kla / 120 - 1]))
        assert not term
        energy += info['energy_kwh']
        mixing += info['mixing_kwh']
        reward += r
        cost += info['cost']
        nh4.extend(info['nh4_samples'])
        do.extend(info['do5_samples'])
        flow.extend(info['effluent_flow_samples'])
    assert trunc
    elapsed = time.perf_counter() - start
    n, d, q = map(np.asarray, (nh4, do, flow))
    return {'weather': weather, 'kla5_per_day': kla, 'substep_minutes': minutes,
            'days': days, 'control_steps': env.T, 'seconds': elapsed,
            'aeration_kwh_per_day': energy / days, 'mixing_kwh_per_day': mixing / days,
            'nh4_time_mean': float(n.mean()), 'nh4_flow_weighted_mean': float(np.average(n, weights=q)),
            'nh4_violation_percent': float((n > 4).mean() * 100),
            'nh4_crossings': int(np.count_nonzero((n[1:] > 4) & (n[:-1] <= 4))),
            'do5_mean': float(d.mean()), 'do5_above_target_percent': float((d > 2).mean() * 100),
            'episode_cost': cost, 'episode_reward': reward}, np.column_stack([n, d, q])


def verify_environment(state):
    env = BSMAerationEnv(state, days=1 / 288)
    ob, _ = env.reset(seed=5)
    first = env.step(np.array([0.]))
    assert first[3] and not first[2]
    try:
        env.step(np.array([0.]))
        raise AssertionError('Missing boundary guard')
    except RuntimeError:
        pass
    np.testing.assert_array_equal(env.reset(seed=5)[0], ob)
    repeated = env.step(np.array([0.]))
    np.testing.assert_array_equal(first[0], repeated[0])
    assert first[1:] == repeated[1:]
    other = BSMAerationEnv(state, days=1 / 288)
    other.reset()
    other.step(np.array([1.]))
    env.reset()
    np.testing.assert_array_equal(env.step(np.array([0.]))[0], first[0])
    env.reset()
    try:
        env.step(np.array([np.nan]))
        raise AssertionError('Missing finite action guard')
    except ValueError:
        pass
    assert env.t == 0
    upper = env.step(np.array([99.]))
    env.reset()
    np.testing.assert_array_equal(upper[0], env.step(np.array([1.]))[0])
    # 같은 초기 상태에서 upstream 5회 직접 적분한 결과와 wrapper를 비교한다.
    raw = make_plant()
    restore(raw, state)
    klas = BASE_KLAS.copy()
    klas[-1] = 120
    energy = 0.
    for i in range(7 * 1440, 7 * 1440 + 5):
        raw.step(i, klas)
        energy += raw.ae / 60
    np.testing.assert_allclose(raw.ys_eff[9], first[4]['nh4_samples'][-1], rtol=0, atol=0)
    np.testing.assert_allclose(energy, first[4]['energy_kwh'], rtol=1e-14)
    np.testing.assert_allclose(first[4]['cost'], np.mean(np.array(first[4]['nh4_samples']) > 4))
    return {'deterministic_reset': True, 'independent_instances': True, 'invalid_action_rejected': True,
            'time_limit_guard': True, 'action_clipping': True, 'direct_upstream_equivalence': True,
            'energy_units_and_substep_cost': True}


def agent_smoke(state):
    import torch
    from fcsrl.utils import DeviceConfig
    from ppo_config import PPOConfig
    from ppo_lag import PPOLagAgent
    from ppo_fcsrl import PPOFCSRLAgent
    from ppo_tail import PPOFCSRLTailAgent, TailConfig
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    # 수치 척도 연결용 5% 시간 예산 예시다. 본 실험 채택 기준이 아니다.
    config = PPOConfig(total_steps=288, n_env=1, horizon=288, n_chunks=1, cost_limit=14.4)
    result = {}
    for name, cls in [('ppo', PPOLagAgent), ('fcsrl_ppo', PPOFCSRLAgent), ('fcsrl_ppo_tail', PPOFCSRLTailAgent)]:
        torch.manual_seed(0)
        extra = {'tail_config': TailConfig(cost_scale=config.cost_limit)} if name.endswith('tail') else {}
        agent = cls(11, 1, config=config, **extra)
        env = BSMAerationEnv(state, days=1 / 24)
        obs, _ = env.reset()
        for _ in range(env.T):
            agent.obs_normalizer.update(obs[None])
            action = np.asarray(agent.act_det(obs)).reshape(1)
            obs, _, _, _, info = env.step(action)
            assert np.isfinite(obs).all()
            if isinstance(agent, PPOFCSRLAgent):
                assert np.isfinite(agent.next_feasibility(agent.obs_normalizer(obs[None]))).all()
        result[name] = {'finite': True, 'transitions': env.T, 'optimizer_updates': 0}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings('error', category=ODEintWarning)
    start = time.perf_counter()
    summary = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'data_archive_sha256': sha(os.environ['BSM1_ARCHIVE']),
        'packages': {x: importlib.metadata.version(x) for x in
                     ('bsm2-python', 'numba', 'llvmlite', 'control', 'numpy', 'scipy', 'torch', 'gymnasium')},
        'source_hashes': {str(p.relative_to(PACKAGE)): sha(p) for p in
                          [PACKAGE/'bsm1_base.py', PACKAGE/'bsm1_ol.py', PACKAGE/'bsm_base.py',
                           PACKAGE/'bsm2/asm1_bsm2.py', PACKAGE/'bsm2/settler1d_bsm2.py',
                           PACKAGE/'data/dryinfluent.csv', PACKAGE/'data/raininfluent.csv']}}
    stable, summary['steady'] = steady(a.out)
    dry = make_plant()
    restore(dry, stable)
    conditioned = condition(dry, 14 * 1440, '14 day dry conditioning')
    snapshots = {}
    for weather in ('dry', 'rain'):
        plant = make_plant(weather)
        restore(plant, conditioned)
        snapshots[weather] = condition(plant, 7 * 1440, f'{weather} 7 day prefix')
        np.savez_compressed(a.out / f'{weather}_initial.npz', **snapshots[weather])
    summary['checks'] = verify_environment(snapshots['dry'])
    summary['baselines'] = []
    for weather in ('dry', 'rain'):
        for kla in (0., 84., 240.):
            row, series = rollout(snapshots[weather], weather, kla)
            summary['baselines'].append(row)
            np.savez_compressed(a.out / f'{weather}_{kla:g}.npz', series=series)
            save(a.out / 'progress.json', summary)
    refined, fine = rollout(snapshots['dry'], 'dry', 84., minutes=.5)
    coarse = np.load(a.out / 'dry_84.npz')['series']
    summary['integration_check'] = {'refined': refined,
        'nh4_endpoint_mae': float(np.abs(fine[1::2, 0] - coarse[:, 0]).mean()),
        'do5_endpoint_mae': float(np.abs(fine[1::2, 1] - coarse[:, 1]).mean())}
    summary['agent_smoke'] = agent_smoke(snapshots['dry'])
    summary['elapsed_seconds'] = time.perf_counter() - start
    summary['status'] = 'preflight_complete_no_rl_training'
    save(a.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
