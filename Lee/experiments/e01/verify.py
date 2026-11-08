"""학습 전 검사. 통과해야 비교 학습을 실행한다.

비용 정의가 의도대로 바뀌었는지, 복사한 Yoo 코드가 원본과 같은지, 하루 경계에서
가치와 feasibility 가 다음 날로 넘어가지 않는지, 실제 최적화가 한 번 도는지 확인한다.
seed 999 는 비교 seed 에 포함하지 않는다.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Lee/code/bsm'))
import bsm_binary as cost_modes  # noqa: E402
from bsm_binary import BinaryEnv, violation_metrics  # noqa: E402
from ppo_fcsrl import feasibility_targets  # noqa: E402
from ppo_lag import gae  # noqa: E402
import train_e01 as train  # noqa: E402

COPIES = {
    'ppo_config.py': 'Yoo/code/wwtp/ppo_config.py',
    'ppo_lag.py': 'Yoo/code/wwtp/ppo_lag.py',
    'ppo_fcsrl.py': 'Yoo/code/wwtp/ppo_fcsrl.py',
    'ppo_tail.py': 'Yoo/code/wwtp/ppo_tail.py',
    'env.py': 'Yoo/code/wwtp/env.py',
    'surrogate.py': 'Yoo/code/wwtp/surrogate.py',
    'FCSRL_LICENSE': 'Yoo/code/wwtp/FCSRL_LICENSE',
    'bsm_env.py': 'Yoo/experiments/e16/bsm_env.py',
    'BSM_LICENSE': 'Yoo/experiments/e16/BSM_LICENSE',
    'bsm_mean.py': 'Yoo/experiments/e17/bsm_mean.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_copies():
    rows = {}
    for name, source in COPIES.items():
        mine, theirs = sha(ROOT / 'Lee/code/bsm' / name), sha(ROOT / source)
        if mine != theirs:
            raise AssertionError(f'{name} differs from {source}')
        rows[name] = mine[:16]
    return rows


def check_module_origin():
    """실제로 가져온 모듈이 Lee 폴더의 복사본인지 확인한다 (경로 선점 방지)."""
    import ppo_config
    import ppo_fcsrl
    import ppo_lag
    import bsm_env
    import bsm_mean

    expected = (ROOT / 'Lee/code/bsm').resolve()
    rows = {}
    for module in (ppo_config, ppo_lag, ppo_fcsrl, bsm_env, bsm_mean, cost_modes):
        path = Path(module.__file__).resolve()
        if path.parent != expected:
            raise AssertionError(f'{module.__name__} was imported from {path}, expected {expected}')
        rows[module.__name__] = str(path.relative_to(ROOT)).replace('\\', '/')
    return rows


def check_cost_definitions(prepared):
    """실제 환경 하루를 돌려 두 비용의 정의와 항등식을 확인한다."""
    initial = dict(np.load(prepared / 'initial.npz'))
    env = BinaryEnv(initial, weather='dry', start_day=0., days=1.)
    env.reset()
    loads, flows, violations, infos = [], [], [], []
    for _ in range(env.T):
        _, _, term, _, info = env.step(np.array([84 / 120 - 1]))
        loads.append(cost_modes.step_cost('mean', info))
        violations.append(cost_modes.step_cost('binary', info))
        flows.append(info['flow_ratio'])
        infos.append(info)
    assert term, 'One day must end as a terminal step'
    loads = np.array(loads)[:, None]
    flows = np.array(flows)[:, None]
    violations = np.array(violations)[:, None]

    metrics = violation_metrics(infos)
    mean_cost = cost_modes.finalize_day('mean', loads, flows)
    identity = float(mean_cost.sum() / 288 * 4)
    assert abs(identity - metrics['flow_weighted_nh4']) < 1e-6, 'Mean cost identity broken'

    binary_cost = cost_modes.finalize_day('binary', violations, flows)
    assert np.array_equal(binary_cost, violations), 'Binary cost must not be rescaled'
    assert set(np.unique(binary_cost)) <= {0.0, 1.0}, 'Binary cost must be 0 or 1'
    expected = np.array([float((x['effluent'][:, 9] > 4).any()) for x in infos])
    assert np.array_equal(binary_cost.ravel(), expected), 'Binary cost must mark instantaneous violations'
    assert abs(binary_cost.sum() - metrics['violating_steps_per_day']) < 1e-9

    # 같은 하루인데 두 비용의 한도 대비 위치가 다르다는 것이 이번 변경의 핵심이다.
    return {'flow_weighted_nh4': metrics['flow_weighted_nh4'],
            'mean_mode_daily_cost': float(mean_cost.sum()), 'mean_mode_limit': cost_modes.MEAN_DAILY_LIMIT,
            'binary_mode_daily_cost': float(binary_cost.sum()), 'binary_mode_limit': cost_modes.BINARY_DAILY_LIMIT,
            'step_violation_percent': metrics['step_violation_percent'],
            'sample_violation_percent': metrics['nh4_violation_percent']}


def check_pid_scaling():
    rows = {}
    for mode in cost_modes.MODES:
        gains = cost_modes.pid_gains(mode)
        limit = cost_modes.daily_limit(mode)
        for name, base in zip(('pid_kp', 'pid_ki', 'pid_kd'), cost_modes.BASE_PID):
            assert abs(gains[name] * limit / (base * cost_modes.BASE_PID_LIMIT) - 1) < 1e-12
        rows[mode] = {'limit': limit, **gains}
    return rows


def check_boundaries():
    """하루 경계에서 GAE 와 feasibility 가 다음 날로 이어지지 않아야 한다."""
    T, N = 6, 1
    end = np.zeros((T, N), np.float32)
    end[2] = end[5] = 1.
    rew = np.ones((T, N), np.float32)
    val = np.zeros((T, N), np.float32)
    next_val = np.zeros((T, N), np.float32)
    adv, _ = gae(rew, val, next_val, end, .99, .95)
    assert abs(adv[2, 0] - 1.) < 1e-6 and abs(adv[5, 0] - 1.) < 1e-6, 'Terminal advantage must not bootstrap'

    cost = np.zeros((T, N), np.float32)
    cost[1] = 1.
    feasi = feasibility_targets(cost, np.zeros((T, N), np.float32), end, end, .9)
    assert abs(feasi[0, 0] - .9) < 1e-6 and abs(feasi[1, 0] - 1.) < 1e-6
    assert feasi[2, 0] == 0. and feasi[3, 0] == 0., 'Feasibility must stop at the day boundary'
    assert feasi.max() <= 1. and feasi.min() >= 0., 'Binary feasibility must stay in [0, 1]'
    return {'terminal_advantage': float(adv[2, 0]), 'feasibility_before_violation': float(feasi[0, 0]),
            'feasibility_after_boundary': float(feasi[2, 0])}


def check_optimization(prepared, updates=1):
    """비교 seed 와 분리된 seed 999 로 실제 갱신 한 번을 수행한다."""
    rows = {}
    observations = np.load(prepared / 'train_observations.npy')
    for mode in cost_modes.MODES:
        for method in ('ppo', 'fcsrl'):
            agent, c = train.make_agent(method, mode, 999, observations, updates)
            c = type(c)(**{**c.to_dict(), 'total_steps': 288 * 2, 'n_env': 2, 'horizon': 288})
            rng = np.random.default_rng(999)
            buf, diag = train.collect(agent, c, mode, prepared, rng)
            info = train.checked_update(agent, buf)
            assert info['policy_parameter_delta_l2'] > 0, 'Policy did not change'
            assert np.isfinite(buf['feasi']).all() and buf['feasi'].min() >= 0
            if mode == 'binary':
                assert buf['feasi'].max() <= 1 + 1e-6, 'Binary feasibility target above 1'
            rows[f'{mode}/{method}'] = {
                'policy_parameter_delta_l2': info['policy_parameter_delta_l2'],
                'full_update_probe_kl': info['full_update_probe_kl'],
                'gradient_norm_max': info['gradient_norm_max'],
                'daily_cost_mean': diag['daily_cost_mean'],
                'feasibility_target_mean': diag['feasibility_target_mean'],
                'feasibility_symlog_outside_fraction': diag['feasibility_symlog_outside_fraction']}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    warnings.filterwarnings('error', category=ODEintWarning)
    torch.set_num_threads(1)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'copied_file_hashes': check_copies(),
              'module_origin': check_module_origin(),
              'pid_scaling': check_pid_scaling(),
              'boundaries': check_boundaries(),
              'cost_definitions': check_cost_definitions(a.prepared),
              'optimization': check_optimization(a.prepared)}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'copied_file_hashes'}, indent=2))
    print('VERIFIED', flush=True)


if __name__ == '__main__':
    main()
