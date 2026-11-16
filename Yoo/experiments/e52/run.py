"""e52 실행: kappa 10 (PPO-Lag, FCSRL) seed 0, 1, 2 와 기준 kappa 0 의 seed 2.

기준의 seed 0, 1 은 같은 코드 경로인 e44 padded 결과를 그대로 쓴다 (kappa 0 에서 e44 학습 기록이 처음 두 갱신의
모든 항목까지 같음을 smoke 로 확인).

    python run.py --out /home/jio/fcsrl_src/results/e52/main --jobs 5
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from common import save, sha  # noqa: E402

PREPARED = Path('/home/jio/fcsrl_src/results/e30/prepared')
CORRECTED = Path('/home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl')
JOBS = [(m, 10.0, s) for s in (0, 1, 2) for m in ('fcsrl', 'ppo')] + [(m, 0.0, 2) for m in ('fcsrl', 'ppo')]


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    repo = HERE.parents[2]
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                'jobs': JOBS, 'input_sha256': {'data.pkl': sha(PREPARED / 'data.pkl'), 'corrected_surrogate.pkl': sha(CORRECTED),
                'prior.pt': sha(PREPARED / 'prior.pt')}, 'source_sha256': {p.name: sha(p) for p in HERE.glob('*.py')}}
    save(a.out / 'manifest.json', manifest)
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=f"{repo / 'Yoo/code/wwtp'}:{HERE.parent / 'e30'}")
    pending, active, done = list(JOBS), [], []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError()))
    try:
        while pending or active:
            while pending and len(active) < a.jobs:
                m, k, s = pending.pop(0)
                out = a.out / 'runs' / m / f'k{k:g}' / f'seed_{s}'
                cmd = [sys.executable, '-u', str(HERE / 'train.py'), '--method', m, '--kappa', str(k), '--seed', str(s),
                       '--prepared', str(PREPARED), '--corrected', str(CORRECTED), '--out', str(out)]
                log = (a.out / f'{m}_k{k:g}_s{s}.log').open('w')
                active.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True), log, m, k, s))
                print(f'[{time.strftime("%H:%M")}] START', m, k, s, flush=True)
            time.sleep(5)
            for item in active.copy():
                p, log, m, k, s = item
                if p.poll() is None:
                    continue
                log.close()
                active.remove(item)
                done.append((m, k, s, p.returncode))
                print(f'[{time.strftime("%H:%M")}] END', m, k, s, p.returncode, f'{len(done)}/{len(JOBS)}', flush=True)
                if p.returncode:
                    raise RuntimeError(f'{m} k{k} s{s} failed')
            save(a.out / 'status.json', {'done': done, 'pending': pending, 'running': [x[2:] for x in active]})
        save(a.out / 'status.json', {'done': done, 'pending': [], 'running': [], 'phase': 'complete'})
    except BaseException:
        for p, log, *_ in active:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        save(a.out / 'status.json', {'done': done, 'phase': 'failed'})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=5)
    main(p.parse_args())
