"""e02 결과 그림

1) tradeoff.png: 방법별 seed 마다의 (하루 공기량, 하루 비용) 점과 실제 운전
2) day_<날짜>.png: 평가 구간 하루에 대해 실제 운전과 학습된 정책의 공기량, NH4, DO 궤적

    python plot.py --runs <결과 폴더> --surrogate <surrogate.json> --out <그림 폴더> [--day 2024-04-15 --seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agents import act_fn, load_agent  # noqa: E402
from env import AIR_MAX, DO_MAX, NH4_MAX, AerationEnv, historical_air  # noqa: E402

COLORS = {"historical": "#555555", "td3": "#d62728", "td3lag": "#1f77b4", "fcsrl": "#2ca02c"}
LABELS = {"historical": "Plant operation", "td3": "TD3 (unconstrained)", "td3lag": "TD3-Lag", "fcsrl": "FCSRL + TD3-Lag"}


def trajectory(env, act_fn):
    obs, _ = env.reset()
    tr, done = [], False
    while not done:
        a = act_fn(obs, env)
        obs, r, term, trunc, info = env.step(a)
        tr.append((info["air"], env.nh4, env.do, env.x["phase"][env.t - 1]))
        done = term or trunc
    return np.array(tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--surrogate", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "avedore" / "aved_5min.pkl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--day", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--cudaid", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # 1) 공기량과 비용
    hist = json.loads((a.runs / "historical" / "historical.json").read_text())["test"]
    lim = None
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(hist["air_m3"] / 1000, hist["cost"], c=COLORS["historical"], marker="*", s=160,
               label=LABELS["historical"], zorder=3)
    for m in ("td3", "td3lag", "fcsrl"):
        pts = [json.loads(p.read_text()) for p in sorted((a.runs / m).glob("seed_*/metrics.json"))]
        if not pts:
            continue
        lim = pts[0]["cost_limit"]
        x = [p["test_best"]["air_m3"] / 1000 for p in pts]
        y = [p["test_best"]["cost"] for p in pts]
        ax.scatter(x, y, c=COLORS[m], label=LABELS[m], alpha=0.8)
    if lim is not None:
        ax.axhline(lim, ls="--", c="k", lw=0.8)
        ax.text(ax.get_xlim()[1], lim, " cost limit", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Air volume (10$^3$ Nm$^3$/day)")
    ax.set_ylabel("Cost (violation steps/day)")
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / "tradeoff.png", dpi=200)

    # 2) 하루 궤적
    env = AerationEnv(a.data, a.surrogate, split="test")
    day = a.day or env.days[len(env.days) // 2]
    env.fixed = [(day, a.member)]
    curves = {"historical": trajectory(env, lambda o, e: [historical_air(e)[e.t]])}
    for m in ("td3lag", "fcsrl"):
        rd = a.runs / m / f"seed_{a.seed}"
        if not (rd / "metrics.json").exists():
            continue
        curves[m] = trajectory(env, act_fn(load_agent(rd, env, a.cudaid)))
    t = np.arange(1, len(curves["historical"]) + 1) * 5 / 60
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    aer = curves["historical"][:, 3] == 2
    for ax in axs:
        ax.fill_between(t, 0, 1, where=aer, transform=ax.get_xaxis_transform(), color="#f3e9c6", lw=0)
    for m, c in curves.items():
        axs[0].plot(t, c[:, 0] / 1000, c=COLORS[m], lw=1, label=LABELS[m])
        axs[1].plot(t, c[:, 1], c=COLORS[m], lw=1)
        axs[2].plot(t, c[:, 2], c=COLORS[m], lw=1)
    axs[1].axhline(NH4_MAX, ls="--", c="k", lw=0.8)
    axs[2].axhline(DO_MAX, ls="--", c="k", lw=0.8)
    axs[0].set_ylabel("Air (10$^3$ Nm$^3$/h)")
    axs[0].set_ylim(0, AIR_MAX / 1000 * 1.05)
    axs[1].set_ylabel("NH$_4$ (mg N/L)")
    axs[2].set_ylabel("DO (mg O$_2$/L)")
    axs[2].set_xlabel(f"Hour of {day} (shaded: aerobic phase)")
    axs[0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out / f"day_{day}.png", dpi=200)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
