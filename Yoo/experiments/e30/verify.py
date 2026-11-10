"""학습 전 데이터 경계, 환경 집계, FCSRL 갱신 및 보정 분리를 검사한다."""
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from common import config, make_agent, METHODS, save
from train import collect
from darrow_env import BatchEnv, Dynamics, load_days, evaluate, all_episodes, AIR_MAX
from surrogate_dw import build_arrays, GBMDynamics, STATE, EXO, ZONES
from fcsrl.utils import DeviceConfig


def main(prepared, out):
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    splits, days = load_days(prepared / 'data.pkl')
    env = BatchEnv(days, Dynamics(prepared / 'surrogate.pkl'))
    # Recorded actions integrate to exactly 95 x 15 minutes, including terminal step.
    ref_air = np.mean([days[d]['air'][:-1].clip(0, AIR_MAX).sum() * .25 for d in splits['train']])
    replay = evaluate(env, splits['train'], lambda _: env.historical_action())
    assert abs(replay['air_m3'] - ref_air) < 1e-7
    assert round(replay['cost']) == 12, replay
    # Reset must not mutate previous terminal info.
    env.reset([(splits['train'][0], 0)])
    for _ in range(95):
        _, reward, _, done, info = env.step([.2])
    old = info['air'].copy()
    assert done
    env.reset([(splits['train'][1], 0)])
    assert np.array_equal(info['air'], old)
    assert np.allclose(reward, -.6)
    c = replace(config(1), n_env=2, horizon=190, total_steps=380, update_epochs=2)
    base = make_agent('fcsrl', 0, prepared, c)
    sample = env.reset([(d, 0) for d in splits['train'][:3]])
    initial = base.act_det(sample)
    buf, _ = collect(base, c, env, splits['train'], np.random.default_rng(0))
    with torch.no_grad():
        lp = base.dist(base._t(buf['obs'])).log_prob(base._t(buf['act'])).sum(-1).numpy()
    assert np.allclose(lp, buf['logp'], atol=1e-5)
    assert buf['end'].sum() == 4 and buf['terminated'].sum() == 0
    rng = torch.get_rng_state().clone()
    base_info = base.update(buf)
    rng_after = torch.get_rng_state().clone()
    report = {'passed': True, 'historical_train_replay': replay, 'terminal_air_preserved': True,
              'raw_action_log_probability': True, 'time_limit_boundaries': True, 'variants': {}}
    for method in METHODS:
        a = make_agent(method, 0, prepared, c)
        assert np.array_equal(a.act_det(sample), initial)
        torch.set_rng_state(rng)
        info = a.update(buf)
        assert all(np.isfinite(v) for v in info.values())
        if method != 'ppo':
            assert torch.equal(rng_after, torch.get_rng_state())
            for name in ('encoder', 'fixed_encoder', 'feasi_head', 'proj_layer', 'post_proj', 'prior'):
                assert all(torch.equal(v, getattr(base, name).state_dict()[k])
                           for k, v in getattr(a, name).state_dict().items()), (method, name)
            assert torch.equal(base.log_std, a.log_std)
        untouched = ('vr', 'vc') if method == 'fcsrl_actor' else ('pi',) if method == 'fcsrl_critic' else ()
        for name in untouched:
            assert all(torch.equal(v, getattr(base, name).state_dict()[k])
                       for k, v in getattr(a, name).state_dict().items())
        if method == 'fcsrl_extra_ppo':
            assert info['extra_attempted_steps'] == 50
            assert info['extra_anchor_kl'] <= c.target_kl + 1e-7
        with tempfile.TemporaryDirectory() as td:
            a.save(td)
            restored = make_agent(method, 8, prepared, c)
            restored.load(td)
            assert np.array_equal(a.act_det(sample), restored.act_det(sample))
        report['variants'][method] = info
    save(out, report)
    print('PASS: environment, raw logp, encoder/EMA/RNG parity, head isolation, save/load', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.prepared, a.out)
