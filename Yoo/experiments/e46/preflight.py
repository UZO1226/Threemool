"""조건별 비용 집계, 원 업데이트 일치, 원 표현 학습과 저장 복원을 검사한다."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from conditional import *
from command import CommandEnv


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    splits, days = load_selected(a.prepared / 'data.pkl')
    definition = group_definition(splits, days)
    altered = deepcopy(days)
    for value in altered.values():
        value['exo'][1:, 1] = -999
    assert group_definition(splits, altered) == definition
    c = config(1)
    base = make_agent('baseline', 0, a.prepared, c, definition)
    candidate = make_agent('conditional', 0, a.prepared, c, definition)
    initial = [deepcopy(m.state_dict()) for m in modules(base)]
    assert all(torch.equal(v, initial[j][k]) for j, m in enumerate(modules(candidate)) for k, v in m.state_dict().items())
    env = TrackedEnv(days, Dynamics(a.corrected), definition)
    rng = torch.get_rng_state()
    buf, diag = collect(base, c, env, splits['train'], np.random.default_rng(0))
    torch.set_rng_state(rng)
    old_buf, old_diag = original_collect(base, c, CommandEnv(days, env.dyn, 'padded'), splits['train'], np.random.default_rng(0))
    assert all(np.array_equal(buf[k], v) for k, v in old_buf.items())
    assert diag['nh4_cost'] == old_diag['nh4_cost']
    assert all(n > 0 for n in diag['group_counts'])
    independent = {g: [] for g in (0, 1)}
    parts = buf['cost_parts'].reshape(c.horizon, c.n_env, 2)
    labels = buf['group'].reshape(c.horizon, c.n_env)
    for lane in range(c.n_env):
        for t in range(0, c.horizon, 95):
            g = int(labels[t, lane])
            assert np.all(labels[t:t + 95, lane] == g)
            independent[g].append(float(parts[t:t + 95, lane, 0].sum()))
    for g in (0, 1):
        assert len(independent[g]) == diag['group_counts'][g]
        assert np.isclose(np.mean(independent[g]), diag['group_nh4_costs'][g])
    x = conditional_advantage(buf['adv_r'], buf['adv_c'], buf['group'], np.array([1.25, 1.25, .5]), definition['probabilities'])
    y = combine_advantage(buf['adv_r'], buf['adv_c'], [1.25, .5])
    np.testing.assert_allclose(x, y, atol=1e-8, rtol=1e-6)
    norm = deepcopy(base.obs_normalizer.state())
    rng = torch.get_rng_state()
    info = base.update(buf)
    torch.set_rng_state(rng)
    candidate_info = candidate.update(buf)
    assert all(np.isfinite(v) for v in candidate_info.values())
    assert all(torch.equal(v, modules(candidate)[j].state_dict()[k]) for j, m in enumerate(modules(base)) for k, v in m.state_dict().items())
    assert torch.equal(base.log_std, candidate.log_std)
    assert any(not torch.equal(v, initial[3][k]) for k, v in candidate.encoder.state_dict().items())
    assert any(not torch.equal(v, initial[4][k]) for k, v in candidate.fixed_encoder.state_dict().items())
    assert all(torch.equal(v, initial[-1][k]) for k, v in candidate.prior.state_dict().items())
    assert norm == candidate.obs_normalizer.state()
    candidate.update_group_constraints(dict(group_nh4_costs=[12., 0.], do_cost=0.))
    assert candidate.group_lambdas[0] > 0 and np.all(candidate.group_lambdas[1:] == 0)
    old = deepcopy(candidate.group_pids[0].__dict__)
    candidate.update_group_constraints(dict(group_nh4_costs=[None, 12.], do_cost=0.))
    assert candidate.group_pids[0].__dict__ == old
    assert candidate.group_lambdas[1] > 0
    candidate.save(a.out / 'checkpoint')
    restored = make_agent('conditional', 0, a.prepared, c, definition)
    restored.load(a.out / 'checkpoint')
    probe = env.reset([(splits['train'][0], 0)])
    np.testing.assert_array_equal(restored.act_det(probe), candidate.act_det(probe))
    np.testing.assert_array_equal(restored.group_lambdas, candidate.group_lambdas)
    # 실제 조건별 승수를 사용한 후속 업데이트와 gradient가 유한한지 검사한다.
    buf2, diag2 = collect(candidate, c, env, splits['train'], np.random.default_rng(1))
    active_info = candidate.update(buf2)
    assert all(np.isfinite(v) for v in active_info.values())
    assert all(torch.isfinite(p).all() for m in modules(candidate) for p in m.parameters())
    summary, _, _, _ = evaluate(env, splits['train'][:1], lambda obs: restored.act_det(obs)[:, 0])
    save(a.out / 'verification.json', dict(passed=True, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         seconds=time.monotonic() - start, groups=definition, input_sha256=input_hashes(a.prepared, a.corrected),
         source_sha256=source_hashes(), future_temperature_invariance=True, initial_weights_equal=True,
         original_collection_exact=True, zero_lambda_update_exact=True, equal_lambda_advantage_equivalence=True,
         group_episodes_independently_checked=64, missing_group_pid_skipped=True, both_encoders_updated=True,
         frozen_prior_and_normalizer=True, checkpoint_restoration_exact=True, active_group_update_finite=True,
         test_evaluated=False, val_evaluated=False, diagnostic_train_day=summary, first_collection=diag))
    print('PREFLIGHT PASS', round(time.monotonic() - start, 2), definition['threshold'], definition['counts'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
