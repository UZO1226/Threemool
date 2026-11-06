"""Yoo/e03 결과 표와 그림

evaluate.py crosseval 이 만든 두 json (학습용 MLP, 평가 전용 GBM) 을 읽어 results.md 와 그림을 만든다.

    python summarize_e03.py --mlp <crosseval_surrogate.json> --gbm <crosseval_surrogate_gbm.json> --out .
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

ORDER = ["td3", "td3lag", "fcsrl", "fcsrl_ucb", "ppolag", "fcsrl_ml5", "fcsrl_ucb_ml5"]
NAMES = {"td3": "TD3 (제약 없음)", "td3lag": "TD3 Lag", "fcsrl": "FCSRL + TD3 Lag", "fcsrl_ucb": "FCSRL + UCB 비용 critic",
         "ppolag": "PPO Lag", "fcsrl_ml5": "FCSRL, lambda 상한 5", "fcsrl_ucb_ml5": "FCSRL + UCB, lambda 상한 5"}
EN = {"td3": "TD3", "td3lag": "TD3-Lag", "fcsrl": "FCSRL", "fcsrl_ucb": "FCSRL+UCB", "ppolag": "PPO-Lag",
      "fcsrl_ml5": "FCSRL (λmax 5)", "fcsrl_ucb_ml5": "FCSRL+UCB (λmax 5)"}
COLORS = {"td3": "#d62728", "td3lag": "#1f77b4", "fcsrl": "#2ca02c", "fcsrl_ucb": "#9467bd", "ppolag": "#ff7f0e",
          "fcsrl_ml5": "#17becf", "fcsrl_ucb_ml5": "#8c564b"}
KEYS = [("air_m3", "공기량 (Nm3/일)", 0), ("nh4_viol_pct", "NH4 위반 (%)", 2), ("do_viol_pct", "DO 위반 (%)", 2),
        ("cost", "비용 (/일)", 1), ("days_within_limit_pct", "한도 만족 에피소드 (%)", 1)]


def validate(d, complete=True):
    seen = set()
    for r in d["runs"]:
        key = (r["tag"], r["seed"])
        if key in seen:
            raise ValueError(f"Duplicate run: {key}")
        seen.add(key)
        if r["eval"]["n_episodes"] != 300:
            raise ValueError(f"Expected 300 evaluation episodes: {key}")
        if not all(np.isfinite(r["eval"][k]) for k, _, _ in KEYS):
            raise ValueError(f"Non-finite metrics: {key}")
    if complete:
        expected = {(tag, seed) for tag in ORDER for seed in range(5)}
        if seen != expected:
            raise ValueError(f"Run set mismatch, missing={expected - seen}, unexpected={seen - expected}")


def final_mlp(mlp):
    """마지막 모델은 학습 직후 저장된 MLP 평가 수치를 쓴다. 가중치 유무와 무관하다."""
    runs = []
    for r in mlp["runs"]:
        final = r.get("train_surrogate_final")
        if final is None:
            final = json.loads((Path(r["run_dir"]) / "metrics.json").read_text())["test_final"]
        runs.append({**r, "eval": final})
    return {"historical": mlp["historical"], "runs": runs}


def group(d):
    by = {}
    for r in d["runs"]:
        by.setdefault(r["tag"], []).append(r["eval"])
    return by


def table(d, title):
    by, h = group(d), d["historical"]
    lines = [f"### {title}", "", "| 방법 | seed 수 | " + " | ".join(k[1] for k in KEYS) + " |",
             "| --- | --- | " + " | ".join("---" for _ in KEYS) + " |",
             "| 실제 운전 (공기량 재생) | 해당 없음 | " + " | ".join(f"{h[k]:.{p}f}" for k, _, p in KEYS) + " |"]
    for t in [x for x in ORDER if x in by]:
        v = by[t]
        cells = [f"{np.mean([r[k] for r in v]):.{p}f} ± {np.std([r[k] for r in v]):.{p}f}" for k, _, p in KEYS]
        lines.append(f"| {NAMES[t]} | {len(v)} | " + " | ".join(cells) + " |")
    return lines + [""]


def tests(d, pairs):
    by = group(d)
    lines = ["| 비교 | 공기량 p | 비용 p | 한도 만족 에피소드 p |", "| --- | --- | --- | --- |"]
    for a, b in pairs:
        if a in by and b in by and len(by[a]) == len(by[b]) == 5:
            p = [ttest_ind([r[k] for r in by[a]], [r[k] for r in by[b]], equal_var=False).pvalue
                 for k in ("air_m3", "cost", "days_within_limit_pct")]
            lines.append(f"| {NAMES[a]} 대 {NAMES[b]} | " + " | ".join(f"{x:.4g}" for x in p) + " |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", type=Path, required=True)
    ap.add_argument("--gbm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final_gbm", type=Path, help="저장된 마지막 체크포인트만 GBM으로 평가한 파일")
    a = ap.parse_args()
    mlp, gbm = json.loads(a.mlp.read_text()), json.loads(a.gbm.read_text())
    validate(mlp)
    validate(gbm)
    pairs = [("fcsrl_ucb", "fcsrl"), ("ppolag", "fcsrl"), ("ppolag", "td3lag"), ("fcsrl_ml5", "fcsrl"),
             ("fcsrl_ucb_ml5", "fcsrl_ucb"), ("fcsrl_ucb_ml5", "fcsrl_ml5"), ("ppolag", "fcsrl_ml5")]
    md = ["값은 평가 구간 300 에피소드(60일 x 구성원 5개), val 기준 선택 모델의 seed 평균 ± 표준편차(ddof=0)다.",
          "한도 만족 비율은 날짜와 모델 구성원 조합에 대한 비율이다. 독립된 300일을 뜻하지 않는다.",
          "검정 단위는 학습 seed 5개이며, Welch t 검정은 다중 비교 보정을 하지 않은 탐색적 결과다.", ""]
    md += table(mlp, "학습용 대리 모델 (MLP) 로 평가")
    md += ["Welch t 검정 (seed 5개, MLP 평가):", ""] + tests(mlp, pairs)
    md += table(gbm, "평가 전용 대리 모델 (GBM) 로 평가")
    md += ["Welch t 검정 (seed 5개, GBM 평가):", ""] + tests(gbm, pairs)
    fm = final_mlp(mlp)
    validate(fm)
    md += table(fm, "마지막 모델의 MLP 평가 (학습 직후 저장된 test_final)")
    md += ["Welch t 검정 (seed 5개, 마지막 모델 MLP 평가):", ""] + tests(fm, pairs)
    if a.final_gbm:
        fg = json.loads(a.final_gbm.read_text())
        validate(fg, complete=False)
        md += table(fg, "마지막 모델의 GBM 평가 (저장된 가중치가 있는 실행만)")
        md += [f"마지막 가중치가 없어 GBM 재평가를 못 한 실행: {len(fg.get('missing_checkpoints', []))}개.",
               "일부 seed만 남은 방법의 결과로 전체 seed 결과를 대신하거나 방법 간 유의성 검정을 하지 않는다.", ""]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.md").write_text("\n".join(md))
    print("\n".join(md))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, d, name in ((axs[0], mlp, "Training surrogate (MLP)"), (axs[1], gbm, "Evaluation-only surrogate (GBM)")):
        h, by = d["historical"], group(d)
        ax.scatter(h["air_m3"] / 1000, h["cost"], marker="*", s=160, c="#555555", label="Plant operation", zorder=3)
        for t in [x for x in ORDER if x in by and x != "td3"]:
            ax.scatter([r["air_m3"] / 1000 for r in by[t]], [r["cost"] for r in by[t]], c=COLORS[t], label=EN[t],
                       alpha=0.8, s=22)
        ax.axhline(15, ls="--", c="k", lw=0.8)
        ax.set_xlabel("Air volume (10$^3$ Nm$^3$/day)")
        ax.set_ylabel("Cost (violation steps/day)")
        ax.set_title(name, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    (a.out / "figures").mkdir(exist_ok=True)
    fig.savefig(a.out / "figures" / "tradeoff_e03.png", dpi=200)


if __name__ == "__main__":
    main()
