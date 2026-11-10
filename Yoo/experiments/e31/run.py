"""e31 실행기: (방법 x 배분 방식 x seed) 를 동시에 몇 개씩 돌리고 전체 진행률을 tqdm 막대 하나로 보여 준다

    python run.py --prepared /home/jio/fcsrl_src/results/e31/prepared --out /home/jio/fcsrl_src/results/e31/stage2 \
        --seeds 0 1 2 --jobs 3
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['ppo', 'fcsrl'])
    p.add_argument('--fracs', nargs='+', default=['recorded', 'fixed'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--jobs', type=int, default=3)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, f, s) for s in a.seeds for f in a.fracs for m in a.methods]
    per = a.updates * 6080
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_WAIT_POLICY='PASSIVE', PYTHONWARNINGS='ignore')
    done = lambda m, f, s: (a.out / f'{m}_{f}' / f'seed_{s}' / 'metrics.json').exists()  # noqa: E731
    pending = [j for j in jobs if not done(*j)]
    finished = len(jobs) - len(pending)
    running, failed = {}, []
    bar = tqdm(total=len(jobs) * per, initial=finished * per, unit='step', desc='e31 2단계', dynamic_ncols=True)
    while pending or running:
        while pending and len(running) < a.jobs:
            m, f, s = pending.pop(0)
            out = a.out / f'{m}_{f}' / f'seed_{s}'
            if out.exists():
                raise SystemExit(f'{out} 가 이미 있다. 지우고 다시 실행한다')
            log = open(a.out / f'{m}_{f}_s{s}.log', 'w')
            cmd = [sys.executable, '-u', str(HERE / 'train.py'), '--method', m, '--frac', f, '--seed', str(s),
                   '--updates', str(a.updates), '--prepared', str(a.prepared), '--out', str(out)]
            running[(m, f, s)] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(15)
        steps = finished * per
        post = []
        for job, (proc, log) in list(running.items()):
            pr = a.out / f'{job[0]}_{job[1]}' / f'seed_{job[2]}' / 'progress.json'
            try:
                q = json.loads(pr.read_text())
            except Exception:
                q = {}
            steps += q.get('steps', 0)
            post.append(f"{job[0]}/{job[1]}/s{job[2]}:{q.get('update', 0)}")
            if proc.poll() is not None:
                log.close()
                del running[job]
                finished += 1
                if proc.returncode != 0 or not done(*job):
                    failed.append(job)
                bar.write(f"[{time.strftime('%H:%M')}] {job} 종료 (코드 {proc.returncode}) {finished}/{len(jobs)}")
        bar.n = min(steps, len(jobs) * per)
        bar.set_postfix_str(' '.join(post))
        bar.refresh()
    bar.close()
    if failed:
        print('실패:', failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
