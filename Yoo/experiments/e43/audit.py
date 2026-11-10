"""제어 피드백, 관측 누락과 날짜 경계를 학습 없이 검사한다."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
YOO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'e30'))
sys.path.insert(0, str(YOO / 'code/wwtp'))
from darrow_env import BatchEnv, Dynamics, STATE, AIR_MAX, all_episodes, evaluate
from darrow_env import load_days
from ppo_lag import gae
from ppo_fcsrl import feasibility_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def past_command_obs(env):
    """후보 관측만 구성한다. 기존 환경, 정책 및 학습기는 변경하지 않는다."""
    return np.column_stack([env.obs(), env.prev_zone / AIR_MAX]).astype(np.float32)


def feedback_check(days, dates, dyn):
    trajectories = []
    for scramble in (False, True):
        selected = {d: deepcopy(days[d]) for d in dates}
        if scramble:
            for x in selected.values():
                x['state'][1:] = 999.
        env = BatchEnv(selected, dyn)
        calls = []

        def act(obs):
            assert np.array_equal(obs[:, :5], env.state.astype(np.float32))
            calls.append(obs.copy())
            return .2 * np.tanh(obs[:, 0] - 4.)

        result = evaluate(env, dates, act)
        assert len(calls) == 95
        assert env.t == 95
        trajectories.append((np.stack(calls), env.state.copy(), result))
    assert np.array_equal(trajectories[0][0], trajectories[1][0])
    assert np.array_equal(trajectories[0][1], trajectories[1][1])
    assert trajectories[0][2] == trajectories[1][2]
    return {'policy_calls_per_episode': 95, 'state_columns_match_each_step': True,
            'future_recorded_water_999_scramble_exactly_invariant': True,
            'policy': 'deterministic diagnostic function, not a trained policy',
            'dates': dates, 'members': dyn.n_members}


def boundary_check():
    reward = np.array([[1.], [2.], [3.], [4.]])
    value = np.zeros_like(reward)
    next_value = np.full_like(reward, 10.)
    end = np.array([[0.], [1.], [0.], [1.]])
    adv, _ = gae(reward, value, next_value, end)
    changed = reward.copy()
    changed[2:] = 999.
    other, _ = gae(changed, value, next_value, end)
    assert np.array_equal(adv[:2], other[:2])
    assert np.isclose(adv[1, 0], 11.9)
    feasi = feasibility_targets(reward, next_value, end, np.zeros_like(end))
    feasi_other = feasibility_targets(changed, next_value, end, np.zeros_like(end))
    assert np.array_equal(feasi[:2], feasi_other[:2])
    assert np.isclose(feasi[1, 0], 9.)
    return {'gae_stops_trace_at_reset': True, 'feasibility_stops_trace_at_reset': True,
            'gae_keeps_truncated_next_value': True, 'feasibility_keeps_truncated_next_value': True,
            'example_last_reward': 2., 'example_next_value': 10.,
            'example_last_advantage': float(adv[1, 0]),
            'example_last_feasibility': float(feasi[1, 0]),
            'gamma': .99, 'last_step_weight': .99 ** 94,
            'feasibility_discount': .9, 'feasibility_last_step_weight': .9 ** 94}


def missing_input_probe(days, dates, dyn):
    """동일 관측에서 숨은 입력만 교체하는 구조 검사. 현실의 인과 효과가 아니다."""
    changes = {k: [] for k in ('current_allocation', 'previous_zone', 'hidden_influent')}
    donor = np.roll(np.arange(len(dates)), 1)
    for t in (1, 48, 94):
        env = BatchEnv(days, dyn)
        env.reset(all_episodes(dates, dyn.n_members))
        env.t = t
        env.state = env.x['state'][:, t].copy()
        env.prev_zone = env.x['zone'][:, t - 1].copy()
        env.prev_air = env.prev_zone.sum(1)
        original_obs = env.obs().copy()
        action = env.historical_action()
        before = past_command_obs(env)
        assert before.shape[1] == 16
        assert np.array_equal(before[:, :11], original_obs)
        # Copy only mutable environment state, sharing immutable model objects.
        def copy_env():
            e = BatchEnv(days, dyn)
            for name in ('episodes', 'x', 'member', 't', 'state', 'prev_zone', 'prev_air'):
                setattr(e, name, deepcopy(getattr(env, name)))
            return e
        base = copy_env()
        base.step(action)
        donor_rows = (donor[:, None] * dyn.n_members + np.arange(dyn.n_members)).ravel()
        for mode in changes:
            alt = copy_env()
            if mode == 'current_allocation':
                alt.x['frac'][:, t] = env.x['frac'][donor_rows, t]
            elif mode == 'previous_zone':
                frac = env.x['frac'][donor_rows, t - 1]
                alt.prev_zone = env.prev_air[:, None] * frac
                assert np.allclose(alt.prev_zone.sum(1), env.prev_air)
                assert not np.array_equal(before, past_command_obs(alt))
            else:
                alt.x['exo'][:, t, 5:] = env.x['exo'][donor_rows, t, 5:]
            assert np.array_equal(original_obs, alt.obs())
            alt.step(action)
            assert np.array_equal(base.prev_air, alt.prev_air)
            changes[mode].append(np.abs(alt.state - base.state))
        # The candidate reads no future water, current allocation or raw current air.
        alt = copy_env()
        alt.x['state'][:, t + 1:] = 999.
        alt.x['frac'][:, t:] = 0.
        alt.x['air'][:, t:] = 999.
        assert np.array_equal(before, past_command_obs(alt))
    result = {}
    for mode, values in changes.items():
        diff = np.concatenate(values)
        result[mode] = {
            'n_member_snapshots': len(diff), 'identical_observation_and_total_action': True,
            'next_state_abs_difference': {s: {'mean': float(diff[:, j].mean()),
                 'p95': float(np.quantile(diff[:, j], .95)), 'max': float(diff[:, j].max()),
                 'nonzero_pct': float(100 * np.mean(diff[:, j] > 1e-10))}
                 for j, s in enumerate(STATE)}}
    return {'dates': dates, 'ticks': [1, 48, 94], 'donor': 'previous train date cyclically, same tick',
            'interpretation': 'Synthetic input sensitivity, not valid counterfactual outcomes or policy improvement.',
            'candidate_observation': {'dim': 16, 'original_11_exactly_preserved': True,
                 'responds_to_previous_zone': True, 'future_water_and_current_allocation_invariant': True},
            'probes': result}


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    splits, days = load_days(a.prepared / 'data.pkl')
    dyn = Dynamics(a.corrected)
    assert [len(splits[k]) for k in ('train', 'val', 'test')] == [188, 40, 41]
    dates = splits['train']
    files = [Path(__file__), HERE.parent / 'e30/darrow_env.py', HERE.parent / 'e30/common.py',
             HERE.parent / 'e33/train.py', YOO / 'code/wwtp/ppo_lag.py', YOO / 'code/wwtp/ppo_fcsrl.py']
    result = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {str(p.relative_to(YOO)): sha(p) for p in files},
              'input_sha256': {str(p): sha(p) for p in [a.prepared / 'data.pkl', a.corrected]},
              'packages': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'feedback': feedback_check(days, dates[:1], dyn),
              'boundary': boundary_check(),
              'missing_inputs': missing_input_probe(days, dates, dyn),
              'new_training': False, 'val_evaluated': False, 'test_evaluated': False}
    result['elapsed_seconds'] = time.monotonic() - start
    (a.out / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'seconds': result['elapsed_seconds'], 'output': str(a.out)}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
