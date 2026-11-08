"""정책 평가와 결과 집계

평가 구간의 모든 날 x 앙상블 구성원 전부를 고정 순서로 한 번씩 돈다.

    # 실제 운전 공기량(기준선) 평가
    python evaluate.py historical --data ... --surrogate ... --out <폴더>
    # 공기량 배율에 대한 대리 모델 반응 확인
    python evaluate.py sensitivity --data ... --surrogate ... --out <폴더>
    # 여러 실행 결과를 표로 모으기
    python evaluate.py summarize --runs <결과 폴더> --out <md 파일>
    # 학습된 정책과 실제 운전을 다른 대리 모델(예: 평가 전용 GBM)로 다시 평가 (Yoo/e03)
    python evaluate.py crosseval --runs <결과 폴더> [<결과 폴더> ...] --data ... --surrogate <모델> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from env import AIR_MAX, DO_MAX, NH4_MAX, STEP_H, AerationEnv, historical_air, load_surrogate
from surrogate import do_inputs, nh4_inputs, phase_onehot


def episodes(env: AerationEnv):
    return [(d, m) for d in env.days for m in range(env.n_members)]


def run_policy(env: AerationEnv, act_fn, n_episodes: int):
    """act_fn(obs, env) -> 행동. 에피소드별 지표 목록을 돌려준다"""
    rows = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        air = rew = cost = vn = vd = n = 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(obs, env))
            done = term or trunc
            air += info["air"] * STEP_H
            rew += r
            cost += info["cost"]
            vn += info["v_nh4"]
            vd += info["v_do"]
            n += 1
        rows.append({"day": env.day, "member": env.member, "air_m3": air, "reward": rew, "cost": cost,
                     "nh4_viol_pct": 100 * vn / n, "do_viol_pct": 100 * vd / n})
    return rows


def summarize_rows(rows, cost_limit):
    a = {k: float(np.mean([r[k] for r in rows])) for k in ("air_m3", "reward", "cost", "nh4_viol_pct", "do_viol_pct")}
    a["days_within_limit_pct"] = float(100 * np.mean([r["cost"] <= cost_limit for r in rows]))
    a["n_episodes"] = len(rows)
    return a


def eval_agent(agent, data, surrogate, split, cost_limit):
    from agents import act_fn

    env = AerationEnv(data, surrogate, split=split, fixed_episodes=None)
    env.fixed = episodes(env)
    agent.train(False)
    rows = run_policy(env, act_fn(agent), len(env.fixed))
    return summarize_rows(rows, cost_limit), rows


def batched_rollout(env: AerationEnv, sur, act_batch, cost_limit, observe=None):
    """env.fixed 의 모든 (날짜, 구성원) 에피소드를 한꺼번에 굴린다. 상태 전이는 sur 로 한다.
    env.step 과 같은 규칙(관측, 공기량 변환, 비용)을 배열로 다시 쓴 것이다. act_batch(obs (N, 11)) -> (N,) [-1, 1]"""
    eps = env.fixed
    X = [env.day_data[d] for d, _ in eps]
    mem = np.array([m for _, m in eps])
    key = lambda k: np.stack([x[k] for x in X])  # (N, T+1)
    phase, temp, q_in, hour, since = key("phase"), key("temp"), key("q_in"), key("hour"), key("since")
    nh4, do = key("nh4")[:, 0].astype(float), key("do")[:, 0].astype(float)
    N, T = len(eps), env.T
    prev = np.zeros(N)
    air_sum, cost, vn, vd = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for t in range(T):
        ph = phase[:, t].astype(int)
        s, c = np.sin(2 * np.pi * hour[:, t] / 24), np.cos(2 * np.pi * hour[:, t] / 24)
        obs = np.column_stack([nh4, do, ph == 0, ph == 1, ph == 2, temp[:, t] / 10, q_in[:, t] / 1000, s, c,
                               np.minimum(since[:, t], 24) / 12, prev / AIR_MAX]).astype(np.float32)
        a = np.clip(np.asarray(act_batch(obs), dtype=float).reshape(N), -1, 1)
        air = np.where(ph == 2, (a + 1) / 2 * AIR_MAX, 0.0)
        oh = phase_onehot(ph)
        do_n, nh4_n = np.empty(N), np.empty(N)
        for m in np.unique(mem):
            i = mem == m
            xd = do_inputs(do[i], air[i], oh[i], temp[i, t], nh4[i], q_in[i, t], s[i], c[i])
            do_n[i] = np.maximum(0.0, do[i] + sur.do_model.predict(xd, m)[0])
            xn = nh4_inputs(nh4[i], do[i], do_n[i], oh[i], temp[i, t], q_in[i, t], s[i], c[i])
            nh4_n[i] = np.maximum(0.0, nh4[i] + sur.nh4_model.predict(xn, m)[0])
        nh4, do, prev = nh4_n, do_n, air
        v1, v2 = (nh4 > NH4_MAX).astype(float), (do > DO_MAX).astype(float)
        vn += v1
        vd += v2
        cost += v1 + v2
        air_sum += air * STEP_H
        if observe is not None:
            observe(t, obs, a, v1 + v2, air)
    if observe is not None:
        ph = phase[:, T].astype(int)
        s, c = np.sin(2 * np.pi * hour[:, T] / 24), np.cos(2 * np.pi * hour[:, T] / 24)
        obs = np.column_stack([nh4, do, ph == 0, ph == 1, ph == 2, temp[:, T] / 10,
                               q_in[:, T] / 1000, s, c, np.minimum(since[:, T], 24) / 12,
                               prev / AIR_MAX]).astype(np.float32)
        observe(T, obs, None, None, None)
    rows = [{"day": d, "member": int(m), "air_m3": float(air_sum[k]), "reward": float(-air_sum[k] / STEP_H / AIR_MAX),
             "cost": float(cost[k]), "nh4_viol_pct": float(100 * vn[k] / T), "do_viol_pct": float(100 * vd[k] / T)}
            for k, (d, m) in enumerate(eps)]
    return summarize_rows(rows, cost_limit), rows


def batch_act_fn(agent):
    """act_fn 의 배치 버전: obs (N, 11) -> (N,)"""
    import torch
    from fcsrl.agent import PPOLagReprAgent
    from fcsrl.data import Batch

    if hasattr(agent, "act_det"):
        return lambda obs: agent.act_det(obs)[:, 0]
    is_ppo = isinstance(agent, PPOLagReprAgent)
    dev = next(agent.actor.parameters()).device

    def act(obs):
        o = agent.obs_normalizer(obs)
        with torch.no_grad():
            if is_ppo:
                ot = torch.as_tensor(o, dtype=torch.float32, device=dev)
                mu, _ = agent.actor(ot, agent.fixed_encoder.zs(ot))
                a = mu.clamp(-1.0, 1.0) * agent._act_scale + agent._act_bias
            else:
                a = agent(Batch(obs=o, info={}), add_noise=False)[0].act
        return a.detach().cpu().numpy()[:, 0]

    return act


def cmd_crosseval(a):
    from agents import load_agent

    sur = load_surrogate(a.surrogate)
    env = AerationEnv(a.data, a.surrogate, split="test")
    env.fixed = episodes(env)
    out = {"surrogate": str(a.surrogate), "checkpoint": a.checkpoint,
           "historical": None, "runs": [], "missing_checkpoints": []}
    # 실제 운전 공기량: 에피소드 순서대로 그날 기록을 넣는다
    hist_air = np.stack([np.clip(env.day_data[d]["air"] / AIR_MAX * 2 - 1, -1, 1) for d, _ in env.fixed])
    step = {"t": 0}

    def hist_act(obs):
        v = hist_air[:, step["t"]]
        step["t"] += 1
        return v

    out["historical"] = batched_rollout(env, sur, hist_act, a.cost_limit)[0]
    for root in a.runs:
        for p in sorted(Path(root).glob("*/seed_*/metrics.json")):
            m = json.loads(p.read_text())
            try:
                agent = load_agent(p.parent, env, a.cudaid, checkpoint=a.checkpoint)
            except FileNotFoundError as exc:
                if a.checkpoint != "final":
                    raise
                out["missing_checkpoints"].append({"run_dir": str(p.parent), "reason": str(exc)})
                continue
            summ, _ = batched_rollout(env, sur, batch_act_fn(agent), a.cost_limit)
            out["runs"].append({"tag": m.get("tag", m["method"]), "method": m["method"], "seed": m["seed"],
                                "run_dir": str(p.parent), "eval": summ,
                                "checkpoint_path": agent.checkpoint_path,
                                "train_surrogate_final": m["test_final"]})
            print(m.get("tag", m["method"]), m["seed"], {k: round(v, 2) for k, v in summ.items()}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix(a.out.suffix + ".tmp")
    temporary.write_text(json.dumps(out, indent=1))
    temporary.replace(a.out)


def cmd_historical(a):
    out = {}
    for split in ("val", "test"):
        env = AerationEnv(a.data, a.surrogate, split=split)
        env.fixed = episodes(env)
        rows = run_policy(env, lambda o, e: [historical_air(e)[e.t]], len(env.fixed))
        out[split] = summarize_rows(rows, a.cost_limit)
        # 대리 모델 없이 실제 측정값으로 센 위반률 (참고용)
        x = [env.day_data[d] for d in env.days]
        out[split]["measured_nh4_viol_pct"] = float(100 * np.mean(np.concatenate([v["nh4"][1:] > 4.0 for v in x])))
        out[split]["measured_do_viol_pct"] = float(100 * np.mean(np.concatenate([v["do"][1:] > 2.0 for v in x])))
        out[split]["measured_air_m3"] = float(np.mean([v["air"][:-1].sum() * STEP_H for v in x]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "historical.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_sensitivity(a):
    """실제 운전 공기량에 배율을 곱해 평가 구간 전체를 돌린다. 대리 모델이 공기량에 맞는 방향으로 반응하는지 본다"""
    env = AerationEnv(a.data, a.surrogate, split="test")
    env.fixed = episodes(env)
    out = {}
    for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        act = lambda o, e, k=k: [np.clip(e.x["air"][e.t] * k / AIR_MAX * 2 - 1, -1, 1)]
        out[str(k)] = summarize_rows(run_policy(env, act, len(env.fixed)), a.cost_limit)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sensitivity.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {x: round(v[x], 2) for x in ("air_m3", "nh4_viol_pct", "do_viol_pct", "cost")})


def cmd_summarize(a):
    runs = sorted(a.runs.glob("*/*/metrics.json"))
    ms = [json.loads(p.read_text()) for p in runs]
    hist = json.loads((a.runs / "historical" / "historical.json").read_text())["test"]
    keys = [("air_m3", "공기량 (Nm3/일)", 0), ("nh4_viol_pct", "NH4 위반 (%)", 2), ("do_viol_pct", "DO 위반 (%)", 2),
            ("cost", "비용 (/일)", 1), ("days_within_limit_pct", "한도 만족 날 (%)", 1)]
    order = ["td3", "td3lag", "fcsrl"]
    names = {"td3": "TD3 (제약 없음)", "td3lag": "TD3 Lag", "fcsrl": "FCSRL + TD3 Lag"}
    titles = {"test_best": "val 기준 최고 epoch 모델", "test_final": "마지막 epoch 모델"}
    out = []
    for which in ("test_best", "test_final"):
        by = {}
        for m in ms:
            by.setdefault(m["method"], []).append(m[which])
        out += [f"### {titles[which]} (test 구간, seed 평균 ± 표준편차)", "",
                "| 방법 | seed 수 | " + " | ".join(k[1] for k in keys) + " |",
                "| --- | --- | " + " | ".join("---" for _ in keys) + " |",
                "| 실제 운전 (공기량 재생) | 해당 없음 | " + " | ".join(f"{hist[k]:.{d}f}" for k, _, d in keys) + " |"]
        for m in [x for x in order if x in by] + [x for x in by if x not in order]:
            v = by[m]
            cells = [f"{np.mean([r[k] for r in v]):.{d}f} ± {np.std([r[k] for r in v]):.{d}f}" for k, _, d in keys]
            out.append(f"| {names.get(m, m)} | {len(v)} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["### seed 별 (val 기준 최고 epoch 모델)", "",
            "| 방법 | seed | 고른 epoch | 공기량 (Nm3/일) | 비용 (/일) | 한도 만족 날 (%) |", "| --- | --- | --- | --- | --- | --- |"]
    for m in sorted(ms, key=lambda m: (order.index(m["method"]) if m["method"] in order else 9, m["seed"])):
        b = m["test_best"]
        out.append(f"| {names.get(m['method'], m['method'])} | {m['seed']} | {m['best_epoch']} | {b['air_m3']:.0f} | "
                   f"{b['cost']:.1f} | {b['days_within_limit_pct']:.1f} |")
    text = "\n".join(out) + "\n"
    a.out.write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("historical")
    h.add_argument("--data", type=Path, required=True)
    h.add_argument("--surrogate", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--cost_limit", type=float, default=15)
    v = sub.add_parser("sensitivity")
    v.add_argument("--data", type=Path, required=True)
    v.add_argument("--surrogate", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--cost_limit", type=float, default=15)
    s = sub.add_parser("summarize")
    s.add_argument("--runs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    x = sub.add_parser("crosseval")
    x.add_argument("--runs", type=Path, nargs="+", required=True)
    x.add_argument("--data", type=Path, required=True)
    x.add_argument("--surrogate", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--cost_limit", type=float, default=15)
    x.add_argument("--cudaid", type=int, default=0)
    x.add_argument("--checkpoint", choices=["best", "final"], default="best")
    a = ap.parse_args()
    {"historical": cmd_historical, "sensitivity": cmd_sensitivity, "summarize": cmd_summarize,
     "crosseval": cmd_crosseval}[a.cmd](a)


if __name__ == "__main__":
    main()
