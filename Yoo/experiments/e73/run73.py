"""e73 실행: check73.py 가 고른 margin 으로 e72 정책 20개와 공기 0 정책을 dev, val 에서 차단과 함께 평가한다.

새 학습은 없다. holdout 과 test 는 읽지 않는다. 시작할 때 코드와 준비물의 sha256 을 manifest 에 적고 끝날 때 확인한다.
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

HERE = Path(__file__).resolve().parent
E68 = HERE.parent / 'e68'
sys.path.insert(0, str(E68))
import env68 as E  # noqa: E402

BASELINES = Path('/home/jio/fcsrl_src/results/e68/main/baselines/baselines.json')
E72 = Path('/home/jio/fcsrl_src/results/e72/main')
FILES = [E68 / n for n in ('env68.py', 'train68.py')] + [HERE / n for n in ('shield73.py', 'eval73.py', 'check73.py', 'protocol.json', 'run73.py', 'report73.py')] + \
        [BASELINES, HERE.parent / 'e30/darrow_env.py', E.PREPARED / 'surrogate.pkl', E.PREPARED / 'data.pkl'] + \
        [E72 / 'runs' / f'{m}_s{s}' / 'model_final.pt' for m in ('fcsrl', 'ppolag') for s in range(10)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    checks = json.loads(a.check.read_text())
    margin = checks['calibration']['chosen_margin']
    assert margin is not None, 'margin 을 고르지 못했다'
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'logs').mkdir()
    hashes = {str(p): sha(p) for p in FILES}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    py = sys.executable
    cmds = []
    for split in ('dev', 'val'):
        cmds.append((f'air0_{split}', [py, '-u', str(HERE / 'eval73.py'), '--policy', 'air0', '--split', split, '--margin', str(margin),
                                       '--out', str(a.out / 'evals' / f'air0_{split}.json')]))
        for s in range(10):
            for m in ('fcsrl', 'ppolag'):
                cmds.append((f'{m}_s{s}_{split}', [py, '-u', str(HERE / 'eval73.py'), '--policy', m, '--seed', str(s), '--split', split,
                                                   '--margin', str(margin), '--out', str(a.out / 'evals' / f'{m}_s{s}_{split}.json')]))
    (a.out / 'manifest.json').write_text(json.dumps(dict(code_commit=commit, margin=margin, check=str(a.check), hashes=hashes, commands=cmds, python=py,
                                                         started_at=datetime.now(timezone.utc).isoformat(), holdout_evaluated=False, test_evaluated=False), indent=1))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
    pending, active, done = list(cmds), [], []

    def status(phase):
        (a.out / 'status.json').write_text(json.dumps(dict(phase=phase, pending=len(pending), running=[n for n, _, _ in active], done=done,
                                                           updated_at=datetime.now(timezone.utc).isoformat()), indent=1))
    while pending or active:
        while pending and len(active) < a.jobs:
            name, cmd = pending.pop(0)
            active.append((name, time.monotonic(), subprocess.Popen(cmd, env=env, stdout=(a.out / 'logs' / f'{name}.log').open('w'),
                                                                     stderr=subprocess.STDOUT, start_new_session=True)))
            status('running')
        for item in list(active):
            name, t0, proc = item
            if proc.poll() is not None:
                active.remove(item)
                done.append(dict(job=name, exit=proc.returncode, seconds=time.monotonic() - t0))
                status('running')
        time.sleep(5)
    failed = [d for d in done if d['exit'] != 0]
    assert all(sha(p) == v for p, v in hashes.items()), '실행 중 파일이 바뀌었다'
    status('failed' if failed else 'complete')
    if not failed:
        subprocess.run([py, str(HERE / 'report73.py'), '--root', str(a.out)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=14)
    main(p.parse_args())
