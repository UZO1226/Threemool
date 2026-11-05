"""FCSRL 원본 재현 실행기 (Yoo/e01)

원본 저장소 scripts/td3_repr_CMDP.py 를 과제와 seed 조합마다 하위 프로세스로 실행하고,
전체 진행률을 tqdm 막대 하나로 보여 준다. 원본 코드는 고치지 않는다.

    python run_repro.py --fcsrl ~/fcsrl_src/FCSRL --out ~/fcsrl_src/results/e01

이미 끝난 조합(.done 표시)은 건너뛰므로 중단 뒤 같은 명령으로 이어서 돌릴 수 있다.
"""
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

TASKS = ["PointGoal1", "PointButton1", "PointPush1", "PointGoal2", "CarGoal1", "CarButton1"]
SEEDS = [100, 200, 300, 400, 500]

STEP_RE = re.compile(r"Epoch #(\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
TEST_RE = re.compile(r"Epoch #(\d+) =+\s*\ntest_reward: ([-\d.]+)±([-\d.]+)\s*\ntest_cost: ([-\d.]+)±([-\d.]+)")


def tail(path: Path, n: int = 4096) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def job_progress(log: Path, epochs: int, steps: int) -> int:
    """로그 끝부분에서 (완료 epoch 수 x steps + 현재 epoch 진행 스텝) 을 읽는다."""
    found = STEP_RE.findall(tail(log).replace("\r", "\n"))
    if not found:
        return 0
    ep, n, _ = map(int, found[-1])
    return min(epochs * steps, (ep - 1) * steps + n)


def test_rows(log: Path):
    text = log.read_text("utf-8", "replace").replace("\r", "\n") if log.exists() else ""
    return [tuple(map(float, m)) for m in TEST_RE.findall(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcsrl", type=Path, required=True, help="원본 FCSRL 저장소 경로")
    ap.add_argument("--out", type=Path, required=True, help="로그, 모델, wandb 저장 위치 (Git 밖)")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--jobs", type=int, default=3, help="동시에 돌릴 실행 수")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    out = args.out.expanduser().resolve()
    fcsrl = args.fcsrl.expanduser().resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(fcsrl / "hyper_params" / "TD3Repr_Lag.yaml"))
    cfg["trainer"]["writer_dir"] = str(out / "log")
    cfg["trainer"]["model_dir"] = str(out / "model")
    hp = out / "TD3Repr_Lag.yaml"
    yaml.safe_dump(cfg, open(hp, "w"))
    epochs, steps = cfg["trainer"]["epoch"], cfg["trainer"]["grad_step_per_epoch"]
    per_job = epochs * steps

    # seed 를 바깥 고리로 두어, 앞쪽 6개가 끝나면 모든 과제에 seed 하나씩 결과가 생기게 한다
    queue = [(t, s) for s, t in itertools.product(args.seeds, args.tasks)]
    name = lambda t, s: f"{t}_s{s}"
    pending = [j for j in queue if not (out / "logs" / f"{name(*j)}.done").exists()]
    done_before = len(queue) - len(pending)

    env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(out), WANDB_SILENT="true",
               PYTHONWARNINGS="ignore")
    running = {}  # job -> (Popen, log, gpu)
    failed = []
    bar = tqdm(total=len(queue) * per_job, initial=done_before * per_job, unit="step",
               desc="FCSRL 재현", dynamic_ncols=True, smoothing=0.05)
    finished = done_before
    slot = itertools.cycle(args.gpus)

    try:
        while pending or running:
            while pending and len(running) < args.jobs:
                t, s = pending.pop(0)
                log = out / "logs" / f"{name(t, s)}.log"
                gpu = next(slot)
                cmd = [sys.executable, "scripts/td3_repr_CMDP.py",
                       "--env_name", f"Safety{t}Gymnasium-v0", "--cudaid", str(gpu),
                       "--seed", str(s), "--hyperparams", str(hp)]
                fh = open(log, "w")
                fh.write(f"# cmd: {' '.join(cmd)}\n# start: {time.strftime('%F %T')}\n")
                fh.flush()
                running[(t, s)] = (subprocess.Popen(cmd, cwd=fcsrl, env=env, stdout=fh,
                                                    stderr=subprocess.STDOUT), log, gpu)

            time.sleep(10)
            prog = {}
            for job, (p, log, gpu) in list(running.items()):
                prog[job] = job_progress(log, epochs, steps)
                if p.poll() is not None:
                    del running[job]
                    finished += 1
                    if p.returncode == 0 and len(test_rows(log)) == epochs:
                        (out / "logs" / f"{name(*job)}.done").write_text(time.strftime("%F %T"))
                    else:
                        failed.append(name(*job))  # 막대 계산에서는 끝난 것으로 친다
                    bar.write(f"[{time.strftime('%H:%M')}] {name(*job)} 종료 "
                              f"(코드 {p.returncode}) {finished}/{len(queue)}")
            bar.n = (finished * per_job
                     + sum(v for j, v in prog.items() if j in running))
            bar.set_postfix_str(" ".join(f"{t}_s{s}:{prog.get((t, s), 0) * 100 // per_job}%"
                                         for (t, s) in running))
            bar.refresh()
    except KeyboardInterrupt:
        for p, _, _ in running.values():
            p.terminate()
        bar.write("중단했습니다. 같은 명령으로 다시 실행하면 끝난 조합은 건너뜁니다.")
    finally:
        bar.close()

    # 요약: 과제와 seed 별 epoch 마다의 시험 결과
    with open(out / "test_by_epoch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seed", "epoch", "test_reward", "reward_std", "test_cost", "cost_std"])
        for t, s in queue:
            for row in test_rows(out / "logs" / f"{name(t, s)}.log"):
                w.writerow([t, s, int(row[0]), *row[1:]])
    if failed:
        print("실패한 조합:", ", ".join(failed))


if __name__ == "__main__":
    main()
