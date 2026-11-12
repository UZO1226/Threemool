"""사전 검사 후 네 학습을 실행하고 실패 시 같은 실행의 작업만 중단한다."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from conditional import HERE, ROOT, save, sha, source_hashes


def main(a):
    verification = json.loads(a.verification.read_text())
    assert verification['passed']
    for category in ('input_sha256', 'source_sha256'):
        for path, h in verification[category].items():
            assert sha(path) == h, path
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/home/jio/fcsrl_src/FCSRL', text=True).strip() == 'd882b91054d52c7a63a18a21f7db67bbbed82b63'
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
         code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         arms=['baseline', 'conditional'], seeds=[0, 1], updates=66, steps_per_run=401280,
         groups=verification['groups'], source_sha256=source_hashes(), input_sha256=verification['input_sha256'],
         verification_sha256=sha(a.verification), python=sys.executable, jobs=2, commands=[], test_evaluated=False)
    pending = [(arm, seed) for seed in (0, 1) for arm in ('baseline', 'conditional')]
    active, completed = [], []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
    def status(phase):
        save(a.out / 'status.json', dict(phase=phase, completed=completed, pending=pending,
             running=[dict(arm=arm, seed=seed, pid=p.pid) for p, f, arm, seed in active],
             elapsed_seconds=time.monotonic() - start, updated_at=datetime.now(timezone.utc).isoformat()))
    def stop(sig, frame):
        raise InterruptedError(sig)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            while pending and len(active) < 2:
                arm, seed = pending.pop(0)
                cmd = [sys.executable, '-u', str(HERE / 'train.py'), '--arm', arm, '--seed', str(seed),
                       '--prepared', str(a.prepared), '--corrected', str(a.corrected),
                       '--out', str(a.out / 'runs' / arm / f'seed_{seed}')]
                log = (a.out / f'{arm}_s{seed}.log').open('w')
                p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((p, log, arm, seed))
                manifest['commands'].append(cmd)
                save(a.out / 'manifest.json', manifest)
                print('START', arm, seed, p.pid, flush=True)
            status('training')
            time.sleep(5)
            for item in active.copy():
                p, log, arm, seed = item
                if p.poll() is None:
                    continue
                log.close()
                active.remove(item)
                completed.append(dict(arm=arm, seed=seed, exit_code=p.returncode))
                if p.returncode:
                    raise RuntimeError(f'{arm} seed{seed} failed')
                print('END', arm, seed, flush=True)
        for category in ('source_sha256', 'input_sha256'):
            for path, digest in manifest[category].items():
                assert sha(path) == digest, path
        status('reporting')
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out)], env=env, check=True)
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(a.out / 'manifest.json', manifest)
        status('complete')
    except BaseException:
        for p, log, arm, seed in active:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            log.close()
        status('failed')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
