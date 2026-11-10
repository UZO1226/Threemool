"""학습 없이 e33 저장 결과를 독립 검산하고 동일 검증 구간의 기준 정책을 평가한다."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e33'))
from dual import config, make_agent, METHODS, save
from common import sha
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes, action_to_air
from fcsrl.utils import DeviceConfig


def metrics(states, air):
    # Arrays: episode, time, state. State samples are AFTER each action.
    nh4 = states[:, :, 0]
    do = states[:, :, 2:]
    n = (nh4 > 4).sum(1)
    d = (do.max(2) > 2).sum(1)
    return {
        'air_m3': float((air.sum(1) * .25).mean()),
        'nh4_cost': float(n.mean()), 'do_cost': float(d.mean()),
        'nh4_viol_pct': float((nh4 > 4).mean() * 100),
        'do_viol_pct': float((do.max(2) > 2).mean() * 100),
        'both_mean_limits': bool(n.mean() <= 6 and d.mean() <= 6),
        'both_limits_episode_pct': float(((n <= 6) & (d <= 6)).mean() * 100),
        'nh4_mean_mg_N_L': float(nh4.mean()),
        'nh4_p95_mg_N_L': float(np.quantile(nh4, .95)),
        'nh4_max_mg_N_L': float(nh4.max()),
        'nh4_excess_integral_mg_N_h_L': float((np.maximum(nh4 - 4, 0).sum(1) * .25).mean()),
        'nh4_episode_count_p95': float(np.quantile(n, .95)),
        'nh4_episode_count_max': int(n.max()),
        'do_zone_means_mg_O2_L': do.mean((0, 1)).tolist(),
        'n_episodes': len(air), 'n_transitions_per_episode': air.shape[1],
    }


def replay(env, days, act):
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    states, airs, rewards = [], [], []
    for _ in range(env.T):
        action = np.asarray(act(obs)).reshape(-1)
        obs, reward, cost, done, info = env.step(action)
        assert np.allclose(info['air'], (np.clip(action, -1, 1) + 1) * 4500)
        assert np.allclose(env.prev_zone.sum(1), info['air'])
        expected = (env.state[:, 0] > 4).astype(int) + (env.state[:, 2:].max(1) > 2)
        assert np.array_equal(cost, expected)
        states.append(env.state.copy())
        airs.append(info['air'].copy())
        rewards.append(reward.copy())
    assert done
    air = np.stack(airs, 1)
    assert np.allclose(np.stack(rewards, 1).sum(1), -air.sum(1) * .25 / 2250)
    return metrics(np.stack(states, 1), air)


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    manifest = json.loads((a.runs / 'manifest.json').read_text())
    assert all(sha(a.prepared / k) == v for k, v in manifest['input_sha256'].items())
    splits, days = load_days(a.prepared / 'data.pkl')
    env = BatchEnv(days, Dynamics(a.prepared / 'surrogate.pkl'))
    assert env.T == 95
    assert np.array_equal(action_to_air([-2, -1, 0, 1, 2]), [0, 0, 4500, 9000, 9000])
    frame = pd.read_pickle(a.prepared / 'data.pkl')['data']
    for day in splits['val']:
        index = frame.loc[day].index
        assert len(index) == 96
        assert np.all(np.diff(index.asi8) == 15 * 60 * 10**9)
    actual = metrics(np.stack([days[d]['state'][1:] for d in splits['val']]),
                     np.stack([days[d]['air'][:-1] for d in splits['val']]))
    rows = [{'method': 'observed_operation', 'kind': 'observed', **actual}]
    hist = replay(env, splits['val'], lambda _: env.historical_action())
    rows.append({'method': 'recorded_action_replay', 'kind': 'surrogate', **hist})
    initial = make_agent('ppo', 0, a.prepared, config())
    bc = replay(env, splits['val'], lambda obs: initial.act_det(obs)[:, 0])
    rows.append({'method': 'initial_BC_policy', 'kind': 'surrogate', **bc})
    print('REFERENCES', json.dumps(rows), flush=True)
    checked = []
    for method in METHODS:
        for seed in manifest['seeds']:
            run = a.runs / 'runs' / method / f'seed_{seed}'
            original = json.loads((run / 'metrics.json').read_text())['val_final']
            episodes = json.loads((run / 'val_episodes.json').read_text())
            for key in ('air_m3', 'nh4_cost', 'do_cost'):
                assert np.isclose(np.mean([e[key] for e in episodes]), original[key], rtol=0, atol=1e-8)
            agent = make_agent(method, seed, a.prepared, config())
            agent.load(run / 'model_final')
            result = replay(env, splits['val'], lambda obs: agent.act_det(obs)[:, 0])
            for key in ('air_m3', 'nh4_cost', 'do_cost', 'both_limits_episode_pct'):
                assert np.isclose(result[key], original[key], rtol=0, atol=1e-6), (method, seed, key)
            assert result['both_mean_limits'] == original['both_mean_limits']
            row = {'method': method, 'seed': seed, 'kind': 'surrogate', **result}
            rows.append(row)
            checked.append({'method': method, 'seed': seed, 'metrics_sha256': sha(run / 'metrics.json'),
                            'episodes_sha256': sha(run / 'val_episodes.json')})
            print(method, seed, json.dumps(result), flush=True)
    for row in rows:
        row['air_change_vs_recorded_pct'] = 100 * (row['air_m3'] / actual['air_m3'] - 1)
        row['air_change_vs_initial_BC_pct'] = 100 * (row['air_m3'] / bc['air_m3'] - 1)
    result = {'audit_script_sha256': sha(Path(__file__)), 'input_sha256': manifest['input_sha256'],
              'checked_saved_runs': checked, 'all_checks_passed': True, 'training_performed': False,
              'test_split_evaluated': False, 'rows': rows,
              'definitions': {'interval_hours': .25, 'transitions': 95, 'episode_hours': 23.75,
                              'air_unit_assumption': 'm3/h; normal reference conditions not verified',
                              'energy_kwh_available': False, 'gamma': config().gamma,
                              'gamma_to_last_transition': config().gamma**94,
                              'six_counts_percent': 6 / 95 * 100,
                              'zero_action_volume': 4500 * 95 * .25,
                              'zero_action_undiscounted_reward': -95 * .5}}
    save(a.out / 'audit.json', result)
    md = ['# e34 지표 검산과 기준 정책 재평가', '',
          '검증 40일, 모델 평가 5구성원(200에피소드). 관측 기록은 40일. 모든 공기량은 23.75시간 부피다.',
          'm³/h 가정이며 표준상태(N) 근거 미확인. 실제 전력량(kWh)은 계산하지 않았다.', '',
          '| 구분 | seed | 공기량 m³/23.75h | 기록 대비 % | 초기 BC 대비 % | NH4 표본 수 | DO 표본 수 | 두 평균 한도 | NH4 초과 적분 mg N h/L |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        md.append(f"| {r['method']} | {r.get('seed', '')} | {r['air_m3']:.2f} | {r['air_change_vs_recorded_pct']:+.2f} | {r['air_change_vs_initial_BC_pct']:+.2f} | {r['nh4_cost']:.3f} | {r['do_cost']:.3f} | {r['both_mean_limits']} | {r['nh4_excess_integral_mg_N_h_L']:.3f} |")
    md += ['', '실측과 대리 모델 수질은 서로 대체할 수 없다. recorded_action_replay의 수질은 모델 예측이다.',
           'NH4 초과 적분은 샘플 기반 직사각형 합으로 농도와 시간의 곱이며 오염물질 질량이 아니다.',
           'e33의 6개 체크포인트를 재생한 공기량, 두 비용, 에피소드 충족률이 저장 결과와 일치했다.',
           '정책별 추가 수질 요약, 입력 및 기존 결과의 SHA256은 audit.json에 있다.', '']
    (a.out / 'results.md').write_text('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
