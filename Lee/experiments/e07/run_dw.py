"""Lee/e07, e08 실행기 (DARROW 환경): (방법, seed) 조합을 병렬로 돌리고 진행률을 기록한다 (Windows 호환).

작업 표기는 `방법:seed` 또는 `ppolag_fcsrl_fpen@가중치:seed` 이다.
    python run_av.py --data aved_5min.pkl --model surrogate.json --upstream <원본 FCSRL> --out <결과> \
        --jobs ppolag_fcsrl:0 ppolag_fcsrl_fpen@0.3:0
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRAIN = ROOT / 'Lee/code/darrow/train_dw.py'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def parse(job):
    """`방법[@벌점가중치][#pid배율]:seed`. 예: ppolag_fcsrl#3:0 (승수 PID 3배)"""
    spec, seed = job.rsplit(':', 1)
    spec, _, pid = spec.partition('#')
    method, _, weight = spec.partition('@')
    label = spec.replace('@', '_w') + (f'_pid{pid}' if pid else '')
    return {'label': label, 'method': method, 'weight': float(weight) if weight else None,
            'pid_mult': float(pid) if pid else None, 'seed': int(seed)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--surrogate', type=Path, required=True)
    p.add_argument('--limit', type=float, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', nargs='+', required=True)
    p.add_argument('--steps', type=int, default=400_000)
    p.add_argument('--evaluations', type=int, default=5)
    p.add_argument('--parallel', type=int, default=6)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.upstream, text=True).strip()
    assert commit == 'd882b91054d52c7a63a18a21f7db67bbbed82b63', 'Unexpected upstream FCSRL commit'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=a.upstream, text=True), 'Upstream FCSRL must be unmodified'
    jobs = [parse(j) for j in a.jobs]
    sources = sorted([*HERE.glob('*.py'), *(ROOT / 'Lee/code/darrow').glob('*.py')])
    save(a.out / 'manifest.json', {
        'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'upstream_commit': commit, 'python': sys.executable, 'jobs': jobs, 'parallel_jobs': a.parallel,
        'steps_per_run': a.steps, 'evaluations': a.evaluations, 'cost_limit': a.limit,
        'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
        'surrogate_sha256': hashlib.sha256(a.surrogate.read_bytes()).hexdigest(),
        'source_hashes': {str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', PYTHONWARNINGS='ignore')
    pending, active, completed = jobs.copy(), [], []
    total = len(jobs) * a.steps
    bar = tqdm(total=total, desc='DARROW transitions', mininterval=10)

    def run_dir(j):
        return a.out / j['label'] / f'seed_{j["seed"]}'

    def status(phase):
        steps = len(completed) * a.steps
        rows = []
        for proc, _, j in active:
            try:
                progress = json.loads((run_dir(j) / 'progress.json').read_text())
            except (OSError, json.JSONDecodeError):
                progress = {'phase': 'starting'}
            steps += progress.get('steps', 0)
            rows.append({'label': j['label'], 'seed': j['seed'], 'pid': proc.pid, **{
                k: v for k, v in progress.items() if k != 'latest'}})
        bar.n = min(steps, total)
        bar.refresh()
        save(a.out / 'status.json', {'phase': phase, 'completed': completed, 'running': rows,
                                     'pending': [(j['label'], j['seed']) for j in pending],
                                     'steps': steps, 'total_steps': total,
                                     'updated_at': datetime.now(timezone.utc).isoformat()})

    try:
        while active or pending:
            while pending and len(active) < a.parallel:
                j = pending.pop(0)
                cmd = [sys.executable, '-u', str(TRAIN), '--method', j['method'], '--seed', str(j['seed']),
                       '--steps', str(a.steps), '--evaluations', str(a.evaluations), '--limit', str(a.limit),
                       '--data', str(a.data.resolve()), '--surrogate', str(a.surrogate.resolve()),
                       '--out', str(run_dir(j).resolve())]
                if j['weight'] is not None:
                    cmd += ['--fpen-weight', str(j['weight'])]
                if j['pid_mult'] is not None:
                    cmd += ['--pid-mult', str(j['pid_mult'])]
                log = (a.out / f'{j["label"]}_s{j["seed"]}.log').open('w')
                proc = subprocess.Popen(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
                active.append((proc, log, j))
                print('START', j['label'], j['seed'], proc.pid, flush=True)
            status('running')
            time.sleep(10)
            for item in active.copy():
                proc, log, j = item
                code = proc.poll()
                if code is None:
                    continue
                active.remove(item)
                log.close()
                if code:
                    raise RuntimeError(f'{j["label"]} seed {j["seed"]} exited {code}')
                completed.append({'label': j['label'], 'seed': j['seed'], 'exit_code': code})
                print('END', j['label'], j['seed'], flush=True)
        status('complete')
        print('ALL COMPLETE', flush=True)
    except BaseException:
        for proc, log, _ in active:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            log.close()
        status('failed')
        raise
    finally:
        bar.close()


if __name__ == '__main__':
    main()
