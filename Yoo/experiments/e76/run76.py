"""e76 실행과 입력 해시, 상태 기록. 정책은 재학습하지 않는다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=1))
    temp.replace(path)


def main(a):
    old = json.loads(Path('/home/jio/fcsrl_src/results/e74/main/manifest.json').read_text())
    checked = {}
    for path, value in old['hashes'].items():
        mapped = str(ROOT / path.split('/사물/', 1)[1]) if '/사물/' in path else path
        assert sha(mapped) == value, f'e74 dependency changed: {mapped}'
        checked[mapped] = value
    assert json.loads(Path('/home/jio/fcsrl_src/results/e76/smoke/smoke.json').read_text())['pass_checks']
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'logs').mkdir()
    files = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md')]
    hashes = dict(checked, **{str(p): sha(p) for p in files})
    for method in ('fcsrl', 'ppolag'):
        for seed in range(10):
            run = Path('/home/jio/fcsrl_src/results/e74/main/runs') / f'{method}_s{seed}'
            for name in ('model_final.pt', 'effective_config.json', 'metrics.json'):
                hashes[str(run / name)] = sha(run / name)
    commands = [(f'{method}_s{seed}', [sys.executable, '-u', str(HERE / 'evaluate76.py'), '--method', method,
                 '--seed', str(seed), '--out', str(a.out / 'jobs' / f'{method}_s{seed}')])
                for seed in range(10) for method in ('fcsrl', 'ppolag')]
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                    hashes=hashes, commands=commands, started_at=datetime.now(timezone.utc).isoformat(),
                    jobs=a.jobs, python=sys.executable, holdout_evaluated=False, test_evaluated=False)
    dump(a.out / 'manifest.json', manifest)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    pending, running, done = list(commands), [], []
    t0 = time.monotonic()
    while pending or running:
        while pending and len(running) < a.jobs:
            name, command = pending.pop(0)
            with (a.out / 'logs' / f'{name}.log').open('w') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, time.monotonic()))
        for name, proc, start in list(running):
            code = proc.poll()
            if code is not None:
                running.remove((name, proc, start))
                done.append(dict(name=name, exit=code, seconds=time.monotonic() - start))
                print(f'{len(done)}/20 {name} exit={code}', flush=True)
        dump(a.out / 'status.json', dict(phase='running', pending=len(pending), running=[n for n, _, _ in running],
                                        done=done, seconds=time.monotonic() - t0))
        if pending or running:
            time.sleep(5)
    assert all(sha(p) == value for p, value in hashes.items()), 'Input changed during run'
    failed = [x for x in done if x['exit'] != 0]
    if not failed:
        subprocess.run([sys.executable, str(HERE / 'report76.py'), '--root', str(a.out)], check=True, env=env)
    dump(a.out / 'status.json', dict(phase='failed' if failed else 'complete', done=done,
                                    seconds=time.monotonic() - t0, hashes_unchanged=True))
    if failed:
        raise RuntimeError(failed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=8)
    main(p.parse_args())
