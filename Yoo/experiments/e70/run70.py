"""e70 실행: e68 학습 코드에 PID 승수 이득을 25 / 0.3 배로 키우고 적분 상한을 묶은 train70.py 로 FCSRL 과 고정 표현 PPO 각 seed 0 ~ 9 를 갱신 250번 학습한다.

비교군은 e68 결과 (같은 dev 날짜, 같은 환경) 를 그대로 쓴다. holdout 과 test 는 읽지 않는다.

시작할 때 코드와 준비물의 sha256 을 manifest 에 적고, 끝날 때 바뀌지 않았는지 확인한다. 실행 중에는 이 폴더 파일을 고치지 않는다.
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
import train68 as T  # noqa: E402

UPDATES = 250
BASELINES = Path('/home/jio/fcsrl_src/results/e68/main/baselines/baselines.json')
FILES = [E68 / n for n in ('env68.py', 'train68.py')] + [HERE / n for n in ('train70.py', 'protocol.json', 'run70.py', 'report70.py')] + [BASELINES] + \
        [HERE.parent / 'e30/darrow_env.py', E.PREPARED / 'surrogate.pkl', E.PREPARED / 'data.pkl',
         T.FCSRL_ROOT / 'fcsrl/agent/ppo_lag_repr.py', T.FCSRL_ROOT / 'fcsrl/utils/solver.py', T.HYPER]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'logs').mkdir()
    hashes = {str(p): sha(p) for p in FILES}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    py = sys.executable
    cmds = []
    for s in range(a.seeds):
        for m in ('fcsrl', 'ppolag'):
            cmds.append((f'{m}_s{s}', [py, '-u', str(HERE / 'train70.py'), '--method', m, '--seed', str(s), '--updates', str(UPDATES),
                                       '--out', str(a.out / 'runs' / f'{m}_s{s}')]))
    (a.out / 'manifest.json').write_text(json.dumps(dict(code_commit=commit, hashes=hashes, commands=cmds, python=py,
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
        time.sleep(10)
    failed = [d for d in done if d['exit'] != 0]
    assert all(sha(p) == v for p, v in hashes.items()), '실행 중 파일이 바뀌었다'
    status('failed' if failed else 'complete')
    if not failed:
        subprocess.run([py, str(HERE / 'report70.py'), '--root', str(a.out)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--jobs', type=int, default=10)
    main(p.parse_args())
