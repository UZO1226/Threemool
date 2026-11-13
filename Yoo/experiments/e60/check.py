"""e60 사전 검사: 차단 없는 수집이 e30 과 정확히 같은지, 차단이 실행 행동에만 들어가는지, 1갱신 진입 검사.

결과 checks.json 의 hashes 를 run.py 가 manifest 로 쓴다.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_shielded as t60  # noqa: E402
se, base, h = t60.se, t60.base, t60.h


def same_buffer(x, y):
    return set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x)


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    c = base.common.config(66)
    splits, days = base.load_selected(se.PREPARED / 'data.pkl')
    dyn = base.Dynamics(se.PREPARED / 'surrogate.pkl')
    env = base.BatchEnv(days, dyn)
    checks = {}
    for method in ('ppo', 'fcsrl'):
        h.configure('cpu')
        agent = base.common.make_agent(method, 0, se.PREPARED, c)
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        buf_e30, diag_e30 = base.e30.collect(agent, c, env, splits['train'], rng)
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        buf_off, diag_off = t60.collect(agent, c, env, splits['train'], rng, None)
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        tick = time.monotonic()
        buf_on, diag_on = t60.collect(agent, c, env, splits['train'], rng, se.make_shield(t60.TRAIN_SHIELD, dyn))
        seconds = time.monotonic() - tick
        exec_air = se.action_to_air(buf_on['act_exec'][:, 0])
        pol_air = se.action_to_air(np.clip(buf_on['act'][:, 0], -1, 1))
        checks[method] = dict(
            shield_off_equals_e30=same_buffer(buf_e30, buf_off) and diag_e30['daily_cost_mean'] == diag_off['daily_cost_mean'],
            # 차단이 상태를 바꾸므로 같은 표본은 첫 시점(t=0, 환경 8개)까지만 기대한다.
            shield_on_same_policy_samples=bool(np.array_equal(buf_on['act'][:c.n_env], buf_off['act'][:c.n_env]) and np.array_equal(buf_on['obs'][:c.n_env], buf_off['obs'][:c.n_env])),
            shield_only_raises_air=bool((exec_air >= pol_air - 1e-9).all()),
            intervention_percent=diag_on['intervention_percent'],
            daily_cost_off=diag_off['daily_cost_mean'], daily_cost_on=diag_on['daily_cost_mean'],
            shield_changes_buffer=not same_buffer(buf_on, buf_off),
            collect_seconds_with_shield=seconds)
    if a.no_smoke:
        checks['smoke'] = 'skipped, see first check run'
    # 1갱신 진입 검사: 실제 실행기와 같은 코드 경로. 결과는 본 실험에 포함하지 않는다.
    smoke = a.out / 'smoke'
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(), hashes=hashes())
    base.save(a.out / 'smoke_manifest.json', manifest)
    for method in (() if a.no_smoke else ('ppo', 'fcsrl')):
        dest = smoke / method
        tick = time.monotonic()
        subprocess.run([sys.executable, str(HERE / 'train_shielded.py'), '--method', method, '--seed', '0', '--updates', '1', '--out', str(dest),
                        '--manifest', str(a.out / 'smoke_manifest.json')], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
                                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        m = json.loads((dest / 'metrics.json').read_text())
        checks[f'smoke_{method}'] = dict(seconds=time.monotonic() - tick, val_none_cost=m['val_none']['cost'],
                                         val_park_e10_cost=m['val_park_e10']['cost'], val_park_e19_cost=m['val_park_e19']['cost'],
                                         train_log_keys=sorted(json.loads((dest / 'train_log.json').read_text())[0]))
    shutil.rmtree(smoke, ignore_errors=True)
    checks['passed'] = all(checks[m]['shield_off_equals_e30'] and checks[m]['shield_on_same_policy_samples'] and checks[m]['shield_only_raises_air']
                           and checks[m]['shield_changes_buffer'] for m in ('ppo', 'fcsrl'))
    checks['hashes'] = hashes()
    base.save(a.out / 'checks.json', checks)
    print(json.dumps({k: v for k, v in checks.items() if k != 'hashes'}, indent=1, ensure_ascii=False))


def hashes():
    files = [se.PREPARED / 'data.pkl', se.PREPARED / 'surrogate.pkl', se.HERE / 'min_brake.py', se.HERE / 'shield_eval.py',
             HERE / 'train_shielded.py', HERE.parent / 'e56/hybrid.py', HERE.parent / 'e56/train.py', HERE.parent / 'e30/train.py',
             HERE.parent / 'e30/common.py', HERE.parent / 'e30/darrow_env.py']
    return {str(p): base.sha(p) for p in files}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-smoke', action='store_true')
    run(p.parse_args())
