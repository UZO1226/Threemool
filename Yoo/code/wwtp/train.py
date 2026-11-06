"""폭기 제어 환경에서 FCSRL + TD3 Lag 와 비교 방법 학습

원본 FCSRL 의 scripts/td3_repr_CMDP.py 흐름을 따르되 다음을 바꿨다.
- 환경: env.AerationEnv (학습은 train 구간, epoch 마다의 시험은 val 구간)
- 병렬 환경마다 다른 seed 를 준다 (원본은 모든 작업 프로세스가 같은 난수 상태를 물려받는다, Yoo/e01 참고)
- 학습이 끝나면 val 기준 최고 epoch 모델과 마지막 모델을 test 구간 전체로 평가해 metrics.json 에 쓴다

방법(--method): agents.py 참고 (fcsrl, td3lag, td3, ppolag, fcsrl_ucb)

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
    ap.add_argument("--method", choices=["fcsrl", "td3lag", "td3", "ppolag", "fcsrl_ucb"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cudaid", type=int, default=0)
    ap.add_argument("--fcsrl_root", type=Path, default=Path.home() / "fcsrl_src" / "FCSRL")
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "avedore" / "aved_5min.pkl")
    ap.add_argument("--surrogate", type=Path, required=True)
    ap.add_argument("--hyperparams", type=Path, default=None, help="없으면 방법에 맞는 설정 (agents.config_path)")
    ap.add_argument("--ucb_n", type=int, default=5, help="fcsrl_ucb: 비용 critic 앙상블 크기")
    ap.add_argument("--ucb_k", type=float, default=0.5, help="fcsrl_ucb: 평균 + k x 표준편차")
    ap.add_argument("--tag", default=None, help="결과 폴더 이름 (없으면 방법 이름)")
    ap.add_argument("--max_lambda", type=float, default=None, help="Lagrange 승수 상한 덮어쓰기")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="TD3 계열 학습 epoch 수")
    ap.add_argument("--grad_steps", type=int, default=None, help="TD3 계열 epoch당 gradient step 수")
    ap.add_argument("--total_steps", type=int, default=None, help="PPO 목표 환경 전이 수")
    ap.add_argument("--eval_chunks", type=int, default=None, help="PPO 학습 중 검증 횟수")
    a = ap.parse_args()
    if a.method == "ppolag" and (a.epochs is not None or a.grad_steps is not None):
        ap.error("PPO에는 --total_steps와 --eval_chunks를 사용하세요. --epochs/--grad_steps는 TD3 전용입니다.")
    if a.method != "ppolag" and (a.total_steps is not None or a.eval_chunks is not None):
        ap.error("--total_steps/--eval_chunks는 PPO 전용입니다.")

    import torch
    import wandb
    from fcsrl.data import Collector, ReplayBuffer
    from fcsrl.env import SubprocVectorEnv
    from fcsrl.trainer import offpolicy_trainer
    from fcsrl.utils import DeviceConfig, dict2attr, set_seed

    from agents import config_path, make_agent
    from env import AerationEnv
    from evaluate import eval_agent

    DeviceConfig().select_device(a.cudaid)
    cfg = yaml.safe_load(open(a.hyperparams or config_path(a.method)))
    run_dir = (a.out / (a.tag or a.method) / f"seed_{a.seed}").resolve()
    if a.method == "ppolag":
        from ppo_config import PPOConfig

        if not isinstance(cfg, dict):
            ap.error("PPO 설정 파일은 key-value mapping이어야 합니다.")
        cfg = dict(cfg)
        for key, value in (("total_steps", a.total_steps), ("n_chunks", a.eval_chunks),
                           ("max_lambda", a.max_lambda)):
            if value is not None:
                cfg[key] = value
        try:
            config = PPOConfig.from_dict(cfg)
        except ValueError as exc:
            ap.error(str(exc))
        return train_ppo(a, config, run_dir)
    if a.epochs:
        cfg["trainer"]["epoch"] = a.epochs
    if a.grad_steps:
        cfg["trainer"]["grad_step_per_epoch"] = a.grad_steps
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["misc"]["seed"] = a.seed
    if a.max_lambda is not None:
        cfg["Lagrangian"]["max_lambda"] = a.max_lambda
    if a.method == "td3":
        # PID 이득을 0 으로 두면 원본 updater 가 KI 로 나누다 멈추므로 상한만 0 으로 막는다
        cfg["Lagrangian"].update(init_lagrg=0.0, max_lambda=0.0)
    make = lambda split, s: (lambda: AerationEnv(a.data, a.surrogate, split=split, seed=s))
    probe = AerationEnv(a.data, a.surrogate, split="train", seed=0)
    e = dict2attr(cfg).env

    base = 1000 * a.seed
    train_envs = SubprocVectorEnv([make("train", base + i) for i in range(e.num_env_train)])
    test_envs = SubprocVectorEnv([make("val", base + 500 + i) for i in range(e.num_env_test)])
    set_seed(a.seed)
    # e02 와 같은 순서: seed 를 정한 뒤 신경망을 만든다
    agent, config = make_agent(a.method, cfg, probe, ucb_n=a.ucb_n, ucb_k=a.ucb_k)

    train_c = Collector(agent, train_envs, ReplayBuffer(config.trainer.replay_size), act_space=probe.action_space)
    test_c = Collector(agent, test_envs)
    wandb.init(project="threemool-wwtp", name=f"{a.tag or a.method}_s{a.seed}", dir=str(run_dir),
               config={"method": a.method, "seed": a.seed, "ucb_n": a.ucb_n, "ucb_k": a.ucb_k, **cfg})
    t = config.trainer
    lag = config.Lagrangian
    info = offpolicy_trainer(agent, train_c, test_c, t.warmup_episode, t.epoch, t.batch_size,
                             t.grad_step_per_epoch, t.collect_len_per_step, t.test_episode,
                             lag.constraint_threshold, None, str(run_dir / "model"), lambda r: False)
    train_c.close()
    test_c.close()

    limit = lag.constraint_threshold
    (run_dir / "model_final").mkdir(exist_ok=True)
    agent.save_model(str(run_dir / "model_final"))
    final, _ = eval_agent(agent, a.data, a.surrogate, "test", limit)
    best_dirs = sorted((run_dir / "model").glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    agent.load_model(str(best_dirs[-1]))  # 트레이너는 val 기준으로 나아질 때만 저장하므로 마지막 저장본이 최고 epoch
    best, rows = eval_agent(agent, a.data, a.surrogate, "test", limit)
    metrics = {"method": a.method, "tag": a.tag or a.method, "seed": a.seed,
               "best_epoch": int(best_dirs[-1].name.split("_")[1]),
               **({"ucb_n": a.ucb_n, "ucb_k": a.ucb_k} if a.method == "fcsrl_ucb" else {}),
               "max_lambda": cfg["Lagrangian"]["max_lambda"],
               "test_best": best, "test_final": final, "cost_limit": limit,
               "train_info": {k: str(v) for k, v in info.items()}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (run_dir / "test_episodes.json").write_text(json.dumps(rows))
    print("TEST_BEST", json.dumps(best))
    print("TEST_FINAL", json.dumps(final))


def train_ppo(a, config, run_dir):
    """ppolag: ppo_lag.py 의 자체 구현 (원본 PPO Lag 는 이 환경에서 학습되지 않았다)"""
    from evaluate import eval_agent
    from fcsrl.utils import DeviceConfig
    from ppo_lag import train

    limit = config.cost_limit
    agent, best = train(a.data, a.surrogate, a.seed, run_dir, device=DeviceConfig.DEVICE, config=config)
    final, _ = eval_agent(agent, a.data, a.surrogate, "test", limit)
    agent.load(run_dir / "model" / f"epoch_{best['epoch']}")
    best_eval, rows = eval_agent(agent, a.data, a.surrogate, "test", limit)
    metrics = {"method": "ppolag", "tag": a.tag or "ppolag", "seed": a.seed, "best_epoch": best["epoch"],
               "max_lambda": config.max_lambda, "impl": "ppo_lag.py", "test_best": best_eval,
               "test_final": final, "cost_limit": limit, "effective_config": config.to_dict(),
               "actual_steps": config.actual_steps, "n_updates": config.n_updates}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (run_dir / "test_episodes.json").write_text(json.dumps(rows))
    print("TEST_BEST", json.dumps(best_eval))
    print("TEST_FINAL", json.dumps(final))


if __name__ == "__main__":
    main()
