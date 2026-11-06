"""Yoo/e02 실행기: 방법 3개 x seed 5개를 동시에 몇 개씩 돌리고 전체 진행률을 tqdm 막대 하나로 보여 준다

    python run_e02.py --surrogate <surrogate.json> --out <결과 폴더> --jobs 4

끝난 조합(metrics.json 있음)은 건너뛰므로 중단 뒤 같은 명령으로 이어서 돌릴 수 있다.
"""
import argparse
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

CODE = Path(__file__).resolve().parents[2] / "code" / "wwtp"
METHODS = ["fcsrl", "td3lag", "td3"]
SEEDS = [0, 1, 2, 3, 4]
STEP_RE = re.compile(r"Epoch #(\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def progress(log: Path, epochs: int, steps: int) -> int:
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            text = f.read().decode("utf-8", "replace").replace("\r", "\n")
    except FileNotFoundError:
        return 0
    found = STEP_RE.findall(text)
    if not found:
        return 0
    ep, n, _ = map(int, found[-1])
    return min(epochs * steps, (ep - 1) * steps + n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    a = ap.parse_args()
    out = a.out.expanduser().resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(CODE / "td3_wwtp.yaml"))
    epochs, steps = cfg["trainer"]["epoch"], cfg["trainer"]["grad_step_per_epoch"]
    per_job = epochs * steps

    queue = [(m, s) for s, m in itertools.product(a.seeds, a.methods)]
    done = lambda m, s: (out / m / f"seed_{s}" / "metrics.json").exists()
    pending = [j for j in queue if not done(*j)]
    finished = len(queue) - len(pending)
    env = dict(os.environ, WANDB_MODE="offline", WANDB_SILENT="true", PYTHONWARNINGS="ignore")
    running, failed = {}, []
    gpu = itertools.cycle(a.gpus)
    bar = tqdm(total=len(queue) * per_job, initial=finished * per_job, unit="step", desc="e02 폭기 제어",
               dynamic_ncols=True, smoothing=0.05)
    try:
        while pending or running:
            while pending and len(running) < a.jobs:
                m, s = pending.pop(0)
                log = out / "logs" / f"{m}_s{s}.log"
                cmd = [sys.executable, str(CODE / "train.py"), "--method", m, "--seed", str(s),
                       "--cudaid", str(next(gpu)), "--surrogate", str(a.surrogate.resolve()), "--out", str(out)]
                fh = open(log, "w")
                fh.write(f"# cmd: {' '.join(cmd)}\n# start: {time.strftime('%F %T')}\n")
                fh.flush()
                running[(m, s)] = (subprocess.Popen(cmd, cwd=CODE, env=env, stdout=fh, stderr=subprocess.STDOUT), log)
            time.sleep(10)
            prog = {}
            for job, (p, log) in list(running.items()):
                prog[job] = progress(log, epochs, steps)
                if p.poll() is not None:
                    del running[job]
                    finished += 1
                    if p.returncode != 0 or not done(*job):
                        failed.append(f"{job[0]}_s{job[1]}")
                    bar.write(f"[{time.strftime('%H:%M')}] {job[0]}_s{job[1]} 종료 (코드 {p.returncode}) "
                              f"{finished}/{len(queue)}")
            bar.n = finished * per_job + sum(v for j, v in prog.items() if j in running)
            bar.set_postfix_str(" ".join(f"{m}_s{s}:{prog.get((m, s), 0) * 100 // per_job}%" for m, s in running))
            bar.refresh()
    except KeyboardInterrupt:
        for p, _ in running.values():
            p.terminate()
        bar.write("중단했습니다. 같은 명령으로 다시 실행하면 끝난 조합은 건너뜁니다.")
    finally:
        bar.close()
    if failed:
        print("실패한 조합:", ", ".join(failed))


if __name__ == "__main__":
    main()
