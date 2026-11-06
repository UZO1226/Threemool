"""정책 평가와 결과 집계

평가 구간의 모든 날 x 앙상블 구성원 전부를 고정 순서로 한 번씩 돈다.

    # 실제 운전 공기량(기준선) 평가
    python evaluate.py historical --data ... --surrogate ... --out <폴더>
    # 공기량 배율에 대한 대리 모델 반응 확인
    python evaluate.py sensitivity --data ... --surrogate ... --out <폴더>
    # 여러 실행 결과를 표로 모으기
    python evaluate.py summarize --runs <결과 폴더> --out <md 파일>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from env import AIR_MAX, STEP_H, AerationEnv, historical_air


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
    import torch
    from fcsrl.data import Batch

    env = AerationEnv(data, surrogate, split=split, fixed_episodes=None)
    env.fixed = episodes(env)
    agent.train(False)

    def act(obs, _env):
        o = agent.obs_normalizer(obs[None])
        with torch.no_grad():
            res, _ = agent(Batch(obs=o, info={}), add_noise=False)
        return res.act.detach().cpu().numpy()[0]

    rows = run_policy(env, act, len(env.fixed))
    return summarize_rows(rows, cost_limit), rows


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
    a = ap.parse_args()
    {"historical": cmd_historical, "sensitivity": cmd_sensitivity, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
