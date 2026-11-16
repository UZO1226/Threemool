"""e66 재개: 서버 재부팅으로 끊긴 같은 test 평가의 남은 작업만 manifest 의 원래 명령 그대로 실행한다.

새 설정이나 정책을 추가하지 않는다. result.json 이 있는 작업은 건드리지 않는다. 끊긴 작업 폴더는 지우지 않고
'<폴더>.interrupted_<날짜>' 로 이름을 바꿔 보존한다. 잠금 파일은 그대로 두고 재개 기록을 RESUME.json 에 남긴다.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
import evaluate66 as e66  # noqa: E402
base = check.base


def main(a):
    assert e66.LOCK.exists(), '원 실행 기록이 없다'
    manifest = json.loads((a.root / 'manifest.json').read_text())
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items()), '원 실행 이후 코드, 모델 또는 정책 파일이 바뀌었다'
    todo, moved = [], []
    stamp = datetime.now().strftime('%Y%m%d')
    for cmd in manifest['commands']:
        out = Path(cmd[cmd.index('--out') + 1])
        if (out / 'result.json').exists():
            continue
        if out.exists():
            dest = out.with_name(out.name + f'.interrupted_{stamp}')
            out.rename(dest)
            moved.append(str(dest))
        todo.append(cmd)
    record = dict(resumed_at=datetime.now(timezone.utc).isoformat(), reason=a.reason, remaining=len(todo), moved=moved,
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip())
    base.save(a.root / 'RESUME.json', record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    status = json.loads((a.root / 'status.json').read_text())
    done = [d for d in status['completed'] if d['exit'] == 0]
    pending, active = list(todo), []

    def save(phase):
        base.save(a.root / 'status.json', dict(phase=phase, pending=len(pending), running=[str(c[c.index('--out') + 1]) for _, c, _ in active],
                                               completed=done, resumed=True, updated_at=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active) < a.jobs:
            cmd = pending.pop(0)
            out = Path(cmd[cmd.index('--out') + 1])
            log = a.root / 'logs' / f'{out.parent.name}_{out.name}.resume.log'
            active.append((time.monotonic(), cmd, subprocess.Popen(cmd, env=env, stdout=log.open('w'), stderr=subprocess.STDOUT, start_new_session=True)))
            save('running')
        for item in list(active):
            t0, cmd, proc = item
            if proc.poll() is not None:
                active.remove(item)
                out = Path(cmd[cmd.index('--out') + 1])
                done.append(dict(job=f'{out.parent.name}/{out.name}', exit=proc.returncode, seconds=time.monotonic() - t0, resumed=True))
                save('running')
        time.sleep(5)
    failed = [d for d in done if d['exit'] != 0]
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items())
    save('failed' if failed else 'complete')
    if not failed:
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.root), '--out', str(HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--jobs', type=int, default=8)
    main(p.parse_args())
