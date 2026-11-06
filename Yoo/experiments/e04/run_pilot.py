"""고정 검증에서 FCSRL 한도 15 대 12를 seed 2개로 비교한다. test 평가는 하지 않는다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[2] / 'code/wwtp'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError(f'Refusing to overwrite {a.out}')
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(f'fcsrl_limit{limit}', seed, limit) for seed in (0, 1) for limit in (15, 12)]
    pending = jobs.copy()
    active, completed = [], []
    env = dict(os.environ, WANDB_MODE='offline', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
        'device': 'cpu', 'parallel_jobs': 2, 'report_cost_limit': 15,
        'inputs': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.data, a.model)},
        'source': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CODE.glob('*.py'))},
        'commands': []}
    def status(phase):
        tmp = a.out / 'status.tmp'
        tmp.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), 'completed': completed,
            'running': [{'tag': tag, 'seed': seed, 'pid': proc.pid} for proc, _, tag, seed in active],
            'pending': pending, 'updated_at': datetime.now(timezone.utc).isoformat()}, indent=2))
        tmp.replace(a.out/'status.json')
    try:
        while pending or active:
            while pending and len(active) < 2:
                tag, seed, limit = pending.pop(0)
                cmd = [sys.executable, '-u', str(CODE/'train.py'), '--method', 'fcsrl', '--tag', tag,
                    '--seed', str(seed), '--cudaid', '-1', '--data', str(a.data), '--surrogate', str(a.model),
                    '--out', str(a.out/'runs'), '--epochs', '5', '--grad_steps', '2500', '--max_lambda', '5',
                    '--cost_limit', str(limit), '--fixed_validation', '--final_eval_split', 'val']
                log = (a.out/f'{tag}_s{seed}.log').open('w')
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((proc, log, tag, seed))
                manifest['commands'].append(cmd)
                (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
                print('START', tag, seed, proc.pid, flush=True)
            status('running')
            time.sleep(10)
            for job in active.copy():
                proc, log, tag, seed = job
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                active.remove(job)
                completed.append({'tag': tag, 'seed': seed, 'exit_code': code})
                print('END', tag, seed, code, flush=True)
                if code:
                    raise RuntimeError(f'{tag} seed {seed} failed ({code})')
        results = []
        for tag, seed, limit in jobs:
            run = a.out/'runs'/tag/f'seed_{seed}'
            m = json.loads((run/'metrics.json').read_text())
            episodes = json.loads((run/'val_episodes.json').read_text())
            pid = json.loads((run/'pid_trace.json').read_text())
            assert m['evaluation_split'] == 'val' and m['validation']['episodes'] == 215
            assert len(episodes) == len({(e['day'],e['member']) for e in episodes}) == 215
            results.append({'tag': tag, 'seed': seed, 'training_limit': limit, 'report_limit': 15,
                'best_epoch': m['best_epoch'], 'val_best': {**m['val_best'],
                    'days_within_limit_pct': float(100*np.mean([e['cost']<=15 for e in episodes]))},
                'final_air': m['val_final']['air_m3'], 'final_cost': m['val_final']['cost'],
                'pid_updates': len(pid), 'lambda_at_cap_fraction': float(np.mean([v['lambda']>=5 for v in pid])),
                'lambda_zero_fraction': float(np.mean([v['lambda']==0 for v in pid])),
                'final_integral': pid[-1]['integral']})
        (a.out/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        status('complete')
    except BaseException:
        for proc, log, _, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        status('failed')
        raise


if __name__ == '__main__':
    main()
