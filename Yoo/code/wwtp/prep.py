"""Avedøre 폭기조 원자료를 5분 간격 표로 정리한다.

    python prep.py --raw Yoo/data/avedore/aved_raw.csv --out Yoo/data/avedore/aved_5min.pkl

- 품질 플래그(quality)가 0 이 아닌 값은 버린다
- 1분 기록을 5분 평균으로 묶고 15분 이하의 빈칸만 선형 보간한다
- 날짜 기준으로 학습, 검증, 평가 구간을 나눈다 (SPLITS)
- 하루(현지 시각 0시부터 288 스텝)의 필수 열이 모두 채워진 날만 에피소드로 쓴다
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T = "BIOLOGY.LINE 3 TANK 1."
COLS = {
    "nh4": T + "NH4",
    "no3": T + "NO3",
    "do": T + "O2",
    "air": T + "Q.AIRFLOW",
    "phase": T + "PROCESSPHASE",
    "temp": T + "TEMPERATURE",
    "do_sp": T + "O2.SETPOINT",
    "q_in": "INLET.Q",
}
REQUIRED = ["nh4", "do", "air", "phase", "temp", "q_in"]  # NO3 는 검증 구간 결측이 많아 상태에서 뺀다
SPLITS = {  # 현지 날짜, 양 끝 포함
    "train": ("2022-09-15", "2023-12-31"),
    "val": ("2024-01-01", "2024-02-29"),
    "test": ("2024-03-01", "2024-06-10"),
}
STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN


def load(raw: Path) -> pd.DataFrame:
    use = ["time"] + [c + " value" for c in COLS.values()] + [c + " quality" for c in COLS.values()]
    df = pd.read_csv(raw, usecols=lambda c: c in use)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Copenhagen")
    out = pd.DataFrame(index=df["time"])
    for k, c in COLS.items():
        v = df[c + " value"].to_numpy(dtype=float)
        qf = df[c + " quality"].to_numpy(dtype=float)
        v[(qf != 0) & ~np.isnan(qf)] = np.nan
        out[k] = v
    return out


def to_5min(x: pd.DataFrame) -> pd.DataFrame:
    r = x.resample(f"{STEP_MIN}min").mean()
    r["phase"] = r["phase"].round()
    r = r.interpolate(limit=3, limit_area="inside")
    r["phase"] = r["phase"].round()
    # 비호기 단계에서는 실제로 공기를 넣지 않는다
    r.loc[r["phase"] != 2, "air"] = 0.0
    r[["nh4", "no3", "do"]] = r[["nh4", "no3", "do"]].clip(lower=0.0)
    return r


def complete_days(r: pd.DataFrame, start: str, end: str):
    days = []
    for day, g in r.loc[start:end].groupby(r.loc[start:end].index.date):
        if len(g) == STEPS_PER_DAY and g[REQUIRED].notna().all().all():
            days.append(str(day))
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = to_5min(load(a.raw))
    meta = {s: complete_days(r, *rng) for s, rng in SPLITS.items()}
    pd.to_pickle({"data": r, "days": meta, "splits": SPLITS}, a.out)
    for s, d in meta.items():
        print(f"{s}: 완전한 날 {len(d)}일 ({SPLITS[s][0]} ~ {SPLITS[s][1]})")


if __name__ == "__main__":
    main()
