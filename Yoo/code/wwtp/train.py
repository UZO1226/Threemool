"""폭기 제어 환경에서 FCSRL + TD3 Lag 와 비교 방법 학습

원본 FCSRL 의 scripts/td3_repr_CMDP.py 흐름을 따르되 다음을 바꿨다.
- 환경: env.AerationEnv (학습은 train 구간, epoch 마다의 시험은 val 구간)
- 병렬 환경마다 다른 seed 를 준다 (원본은 모든 작업 프로세스가 같은 난수 상태를 물려받는다, Yoo/e01 참고)
- 학습이 끝나면 val 기준 최고 epoch 모델과 마지막 모델을 test 구간 전체로 평가해 metrics.json 에 쓴다

방법(--method):
- fcsrl : 원본 FCSRL + TD3 Lag
- td3lag: 같은 신경망에서 표현 학습(encoder 학습)만 끈 TD3 Lag
- td3   : td3lag 에서 Lagrangian 까지 끈 제약 없는 TD3 (lambda = 0)

    python train.py --method fcsrl --seed 0 --cudaid 0 --out <결과 폴더>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fcsrl", "td3lag", "td3"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cudaid", type=int, default=0)
    ap.add_argument("--fcsrl_root", type=Path, default=Path.home() / "fcsrl_src" / "FCSRL")
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "avedore" / "aved_5min.pkl")
    ap.add_argument("--surrogate", type=Path, required=True)
    ap.add_argument("--hyperparams", type=Path, default=HERE / "td3_wwtp.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="설정 파일 값을 덮어쓴다 (시험용)")
    ap.add_argument("--grad_steps", type=int, default=None)
    a = ap.parse_args()

    import torch
    import wandb
    from fcsrl.agent import TD3LagReprAgent
    from fcsrl.data import Collector, ReplayBuffer
    from fcsrl.env import SubprocVectorEnv
    from fcsrl.trainer import offpolicy_trainer
    from fcsrl.utils import DeviceConfig, MeanStdNormalizer, dict2attr, set_seed

    from env import AerationEnv
    from evaluate import eval_agent

    DeviceConfig().select_device(a.cudaid)
    cfg = yaml.safe_load(open(a.hyperparams))
    if a.epochs:
        cfg["trainer"]["epoch"] = a.epochs
    if a.grad_steps:
        cfg["trainer"]["grad_step_per_epoch"] = a.grad_steps
    run_dir = (a.out / a.method / f"seed_{a.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["misc"]["seed"] = a.seed
    if a.method == "td3":
        # PID 이득을 0 으로 두면 원본 updater 가 KI 로 나누다 멈추므로 상한만 0 으로 막는다
        cfg["Lagrangian"].update(init_lagrg=0.0, max_lambda=0.0)
    config = dict2attr(cfg)
    e = config.env

    make = lambda split, s: (lambda: AerationEnv(a.data, a.surrogate, split=split, seed=s))
    probe = AerationEnv(a.data, a.surrogate, split="train", seed=0)
    config.network.s_dim = int(np.prod(probe.observation_space.shape))
    config.network.a_dim = int(np.prod(probe.action_space.shape))
    config.agent.action_range = [float(probe.action_space.low[0]), float(probe.action_space.high[0])]

    base = 1000 * a.seed
    train_envs = SubprocVectorEnv([make("train", base + i) for i in range(e.num_env_train)])
    test_envs = SubprocVectorEnv([make("val", base + 500 + i) for i in range(e.num_env_test)])
    set_seed(a.seed)

    agent = TD3LagReprAgent(config, obs_normalizer=MeanStdNormalizer())
    if a.method in ("td3lag", "td3"):
        agent.train_encoder = lambda batch: {}  # 표현 학습 끄기. encoder 는 초기값 그대로 고정된다

    train_c = Collector(agent, train_envs, ReplayBuffer(config.trainer.replay_size), act_space=probe.action_space)
    test_c = Collector(agent, test_envs)
    wandb.init(project="threemool-wwtp", name=f"{a.method}_s{a.seed}", dir=str(run_dir),
               config={"method": a.method, "seed": a.seed, **cfg})
    t = config.trainer
    lag = config.Lagrangian
    info = offpolicy_trainer(agent, train_c, test_c, t.warmup_episode, t.epoch, t.batch_size,
                             t.grad_step_per_epoch, t.collect_len_per_step, t.test_episode,
                             lag.constraint_threshold, None, str(run_dir / "model"), lambda r: False)
    train_c.close()
    test_c.close()

    limit = lag.constraint_threshold
    final, _ = eval_agent(agent, a.data, a.surrogate, "test", limit)
    best_dirs = sorted((run_dir / "model").glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    agent.load_model(str(best_dirs[-1]))  # 트레이너는 val 기준으로 나아질 때만 저장하므로 마지막 저장본이 최고 epoch
    best, rows = eval_agent(agent, a.data, a.surrogate, "test", limit)
    metrics = {"method": a.method, "seed": a.seed, "best_epoch": int(best_dirs[-1].name.split("_")[1]),
               "test_best": best, "test_final": final, "cost_limit": limit,
               "train_info": {k: str(v) for k, v in info.items()}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (run_dir / "test_episodes.json").write_text(json.dumps(rows))
    print("TEST_BEST", json.dumps(best))
    print("TEST_FINAL", json.dumps(final))


if __name__ == "__main__":
    main()
