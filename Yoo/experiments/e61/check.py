"""e61 사전 검사.

A. 설정: 바꾼 설정은 cost_limit 만 다르고 나머지는 e60 설정과 같다.
B. 재현: 한도 12 로 e60 학습 고리를 2갱신 돌리면 e60 seed 0 의 train_log 2개 기록이 시간 항목을 빼고 정확히 같다.
   한도 2 로 돌리면 첫 갱신 기록은 승수 말고 같고, 승수는 0 보다 크며, 두 번째 갱신은 달라진다.
C. 진입점: 실제 명령줄로 1갱신 학습, 저장, 복원, 다섯 평가를 끝까지 돌리고 effective_config 의 한도와
   첫 갱신 승수가 B 와 같은지 본다. 이 결과는 본 실험에 쓰지 않는다.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_limit as tl  # noqa: E402
t60 = tl.t60
se, base, h = t60.se, t60.base, t60.h
E60 = {'ppo': Path('/home/jio/fcsrl_src/results/e60/main/runs/ppo/seed_0'), 'fcsrl': Path('/home/jio/fcsrl_src/results/e60/main/runs/fcsrl/seed_0')}
TIMING = {'collection_seconds', 'update_seconds'}


def loop(method, seed, limit, updates):
    """t60.run 의 학습 고리와 같은 계산. 평가와 저장은 하지 않는다."""
    tl.common.config = tl.ORIGINAL_CONFIG if limit == 12 else tl.install(limit)
    c = tl.common.config(66)
    device = 'cuda:0' if method == 'fcsrl' else 'cpu'
    learner, collector = h.make_pair(method, seed, se.PREPARED, c, 'on', device)
    rng = np.random.default_rng(seed)
    splits, days = base.load_selected(se.PREPARED / 'data.pkl')
    dyn = base.Dynamics(se.PREPARED / 'surrogate.pkl')
    env = base.BatchEnv(days, dyn)
    shield = se.make_shield(t60.TRAIN_SHIELD, dyn)
    records = []
    for u in range(1, updates + 1):
        h.sync_collector(learner, collector)
        h.configure('cpu')
        buf, diag = t60.collect(collector, c, env, splits['train'], rng, shield)
        info = h.update(learner, buf)
        learner.pid.update(diag['daily_cost_mean'], c.cost_limit)
        learner.lagrg = learner.pid.get_lagrg()
        records.append(dict(update=u, **{'lambda': learner.lagrg}, **diag, **info))
    tl.common.config = tl.ORIGINAL_CONFIG
    return c, records


def same(a, b, skip=()):
    keys = (set(a) | set(b)) - TIMING - set(skip)
    return all(k in a and k in b and a[k] == b[k] for k in keys), sorted(k for k in keys if a.get(k) != b.get(k))


def hashes():
    files = [se.PREPARED / 'data.pkl', se.PREPARED / 'surrogate.pkl', se.HERE / 'min_brake.py', se.HERE / 'shield_eval.py',
             HERE.parent / 'e60/train_shielded.py', HERE / 'train_limit.py', HERE.parent / 'e56/hybrid.py', HERE.parent / 'e56/train.py',
             HERE.parent / 'e30/train.py', HERE.parent / 'e30/common.py', HERE.parent / 'e30/darrow_env.py']
    return {str(p): base.sha(p) for p in files}


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    checks = {}
    c12 = tl.ORIGINAL_CONFIG(66)
    c2 = tl.install(a.limit)(66)
    tl.common.config = tl.ORIGINAL_CONFIG
    checks['config'] = dict(original_limit=c12.cost_limit, new_limit=c2.cost_limit,
                            others_equal={k: v for k, v in c2.to_dict().items() if k != 'cost_limit'} == {k: v for k, v in c12.to_dict().items() if k != 'cost_limit'},
                            e60_config_equal=json.loads((E60['ppo'] / 'effective_config.json').read_text())['config'] == c12.to_dict())
    for method in ('ppo', 'fcsrl'):
        e60_log = json.loads((E60[method] / 'train_log.json').read_text())
        tick = time.monotonic()
        _, r12 = loop(method, 0, 12, 2)
        _, r2 = loop(method, 0, a.limit, 2)
        ok12 = [same(r12[i], e60_log[i]) for i in range(2)]
        ok2 = same(r2[0], e60_log[0], skip=('lambda',))
        checks[method] = dict(
            limit12_reproduces_e60_u1=ok12[0][0], limit12_reproduces_e60_u2=ok12[1][0], limit12_diff_keys=[ok12[0][1], ok12[1][1]],
            limit2_u1_same_except_lambda=ok2[0], limit2_u1_diff_keys=ok2[1],
            limit2_lambda_u1=r2[0]['lambda'], limit2_lambda_u2=r2[1]['lambda'], e60_lambda_u1=e60_log[0]['lambda'],
            daily_cost_u1=r2[0]['daily_cost_mean'], limit2_u2_differs=r2[1].get('kl') != e60_log[1].get('kl'),
            seconds=time.monotonic() - tick)
    # C. 진입점
    smoke = a.out / 'smoke'
    base.save(a.out / 'smoke_manifest.json', dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                                                   hashes=hashes()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    procs = {}
    for method in ('ppo', 'fcsrl'):
        log = (a.out / f'smoke_{method}.log').open('w')
        procs[method] = (time.monotonic(), subprocess.Popen(
            [sys.executable, str(HERE / 'train_limit.py'), '--train-cost-limit', str(a.limit), '--method', method, '--seed', '0', '--updates', '1',
             '--out', str(smoke / method), '--manifest', str(a.out / 'smoke_manifest.json')], env=env, stdout=log, stderr=subprocess.STDOUT))
    for method, (start, proc) in procs.items():
        code = proc.wait()
        entry = dict(exit=code, seconds=time.monotonic() - start)
        if code == 0:
            cfg = json.loads((smoke / method / 'effective_config.json').read_text())
            log = json.loads((smoke / method / 'train_log.json').read_text())
            m = json.loads((smoke / method / 'metrics.json').read_text())
            entry.update(config_limit=cfg['config']['cost_limit'], lambda_u1=log[0]['lambda'],
                         lambda_matches_loop=log[0]['lambda'] == checks[method]['limit2_lambda_u1'],
                         evaluations=sorted(k for k in m if k.startswith(('val_', 'train_')) and k != 'val_final'))
        checks[f'entry_{method}'] = entry
    shutil.rmtree(smoke, ignore_errors=True)
    checks['passed'] = bool(
        checks['config']['others_equal'] and checks['config']['e60_config_equal'] and checks['config']['new_limit'] == a.limit
        and all(checks[m]['limit12_reproduces_e60_u1'] and checks[m]['limit12_reproduces_e60_u2'] and checks[m]['limit2_u1_same_except_lambda']
                and checks[m]['limit2_lambda_u1'] > 0 and checks[m]['e60_lambda_u1'] == 0 and checks[m]['limit2_u2_differs'] for m in ('ppo', 'fcsrl'))
        and all(checks[f'entry_{m}']['exit'] == 0 and checks[f'entry_{m}']['config_limit'] == a.limit and checks[f'entry_{m}']['lambda_matches_loop']
                and len(checks[f'entry_{m}']['evaluations']) == 5 for m in ('ppo', 'fcsrl')))
    checks['limit'] = a.limit
    checks['hashes'] = hashes()
    base.save(a.out / 'checks.json', checks)
    print(json.dumps({k: v for k, v in checks.items() if k != 'hashes'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=float, default=2.0)
    run(p.parse_args())
