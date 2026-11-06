"""Yoo/e01 진단: 저장된 모델을 서로 다른 맵 여러 개로 다시 평가한다

원본 학습 중 시험은 병렬 환경이 모두 같은 맵이라 epoch 하나의 시험 값이 사실상 에피소드 하나다.
여기서는 run 마다 트레이너가 마지막으로 저장한 모델(원본 규칙의 최고 epoch)을 불러
reset(seed=k) 로 맵을 바꿔 가며 에피소드 n 개를 돌린다. 탐색 잡음은 끈다(원본 시험과 같음).

    python reeval.py --fcsrl /home/jio/fcsrl_src/FCSRL --models /home/jio/fcsrl_src/results/e01/model \
        --out reeval.csv --episodes 50 --workers 6
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

TASKS = ["PointGoal1", "PointButton1", "PointPush1", "PointGoal2", "CarGoal1", "CarButton1"]
SEEDS = [100, 200, 300, 400, 500]


def run(job):
    fcsrl, models, task, seed, n_ep, base = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, fcsrl)
    import gymnasium as gym
    import numpy as np
    import safety_gymnasium  # noqa: F401  (환경 등록)
    import torch
    import yaml
    from fcsrl.agent import TD3LagReprAgent
    from fcsrl.data import Batch
    from fcsrl.env import ActionRepeatWrapper, GoalWrapper
    from fcsrl.utils import DeviceConfig, MeanStdNormalizer, dict2attr

    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    name = f"Safety{task}Gymnasium-v0"
    cfg = yaml.safe_load(open(Path(fcsrl) / "hyper_params" / "TD3Repr_Lag.yaml"))
    cfg["env"]["name"] = name
    cfg["network"]["repr_type"] = "FCSRL"
    config = dict2attr(cfg)
    probe = GoalWrapper(gym.make(name))
    config.network.s_dim = int(np.prod(probe.observation_space.shape))
    config.network.a_dim = int(np.prod(probe.action_space.shape))
    config.agent.action_range = [probe.action_space.low[0], probe.action_space.high[0]]
    agent = TD3LagReprAgent(config, obs_normalizer=MeanStdNormalizer())
    run_dir = Path(models) / name / f"td3lag_FCSRL_seed_{seed}"
    ep = max(int(p.name.split("_")[1]) for p in run_dir.glob("epoch_*"))
    agent.load_model(str(run_dir / f"epoch_{ep}"))
    agent.train(False)
    agent.obs_normalizer.set_read_only()

    env = ActionRepeatWrapper(gym.make(name), 4)  # 원본 시험 환경과 같은 구성
    rows = []
    for i in range(n_ep):
        obs, _ = env.reset(seed=base + i)
        done, rew, cost = False, 0.0, 0.0
        while not done:
            o = agent.obs_normalizer(np.array([obs]))
            with torch.no_grad():
                res, _ = agent(Batch(obs=o), add_noise=False)
            obs, r, term, trunc, info = env.step(res.act.cpu().numpy()[0])
            rew += r
            cost += info["cost"]
            done = term or trunc
        rows.append({"task": task, "seed": seed, "epoch": ep, "map_seed": base + i, "reward": rew, "cost": cost})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcsrl", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    a = ap.parse_args()
    # 모든 run 이 같은 맵 묶음(seed 10000 부터)을 보도록 맞춘다
    jobs = [(a.fcsrl, a.models, t, s, a.episodes, 10000) for t in a.tasks for s in SEEDS]
    from tqdm import tqdm
    out = []
    with Pool(a.workers) as pool:
        for rows in tqdm(pool.imap_unordered(run, jobs), total=len(jobs), desc="e01 재평가"):
            out += rows
    out.sort(key=lambda r: (r["task"], r["seed"], r["map_seed"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
