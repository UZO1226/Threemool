"""e18 실행기: 방법 x seed 를 동시에 돌리고 전체 진행률을 tqdm 막대 하나로 보여 준다

    python run.py --prepared <준비 폴더> --out <결과 폴더> [--methods ppo fcsrl --seeds 0 1 --updates 40]
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def save(path, x):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['ppo', 'fcsrl'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--tanks', nargs='+', type=int, default=[5])
    p.add_argument('--pid_mult', type=float, default=1.)
    p.add_argument('--desc', default='e18 학습 전이')
    p.add_argument('--continuous_days', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    per_run = a.updates * 4 * (a.continuous_days * 288 if a.continuous_days else 1440)
    jobs = [(m, s) for s in a.seeds for m in a.methods]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               OMP_WAIT_POLICY='PASSIVE', PYTHONWARNINGS='ignore')
    save(a.out / 'manifest.json', {'started_at': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
        'updates': a.updates, 'steps_per_run': per_run, 'tanks': a.tanks, 'pid_mult': a.pid_mult, 'continuous_days': a.continuous_days,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    pending, active, done = jobs.copy(), [], []
    bar = tqdm(total=len(jobs) * per_run, desc=a.desc, unit='step', mininterval=5, dynamic_ncols=True)

    def stop(signum, frame):
        raise InterruptedError(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            while pending and len(active) < a.jobs:
                m, s = pending.pop(0)
                out = (a.out / m / f'seed_{s}').resolve()
                cmd = [sys.executable, '-u', str(HERE / 'train.py'), '--method', m, '--seed', str(s),
                       '--prepared', str(a.prepared.resolve()), '--out', str(out), '--updates', str(a.updates),
                       '--tanks', *map(str, a.tanks), '--pid_mult', str(a.pid_mult),
                       '--continuous_days', str(a.continuous_days)]
                log = (a.out / f'{m}_s{s}.log').open('w')
                active.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True), log, m, s))
            time.sleep(10)
            steps = len(done) * per_run
            post = []
            for item in active.copy():
                proc, log, m, s = item
                prog = a.out / m / f'seed_{s}' / 'progress.json'
                pr = json.loads(prog.read_text()) if prog.exists() else {}
                steps += pr.get('steps', 0)
                lat = pr.get('latest', {})
                post.append(f"{m}{s}:u{pr.get('update', 0)} lam{lat.get('lambda', 0):.2f}")
                code = proc.poll()
                if code is None:
                    continue
                active.remove(item)
                log.close()
                if code:
                    raise RuntimeError(f'{m} seed {s} exited {code}')
                done.append((m, s))
                bar.write(f'[{time.strftime("%H:%M")}] {m} seed {s} 완료 ({len(done)}/{len(jobs)})')
            bar.n = steps
            bar.set_postfix_str(' '.join(post))
            bar.refresh()
        save(a.out / 'status.json', {'phase': 'complete', 'done': done})
    except BaseException:
        for proc, log, _, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            log.close()
        save(a.out / 'status.json', {'phase': 'failed', 'done': done})
        raise
    finally:
        bar.close()


if __name__ == '__main__':
    main()
