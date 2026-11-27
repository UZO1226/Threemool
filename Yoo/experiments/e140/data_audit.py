"""Inventory the distributed files and characterize the later period's logged inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import openpyxl
import pandas as pd


def run(raw, output):
    inventory = {}
    for path in sorted(raw.iterdir()):
        if path.suffix not in [".xlsx", ".csv"]:
            continue
        entry = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix == ".xlsx":
            book = openpyxl.load_workbook(path, read_only=True)
            entry["sheets"] = {s.title: list(next(s.values)) for s in book}
            book.close()
        else:
            entry["columns"] = list(pd.read_csv(path, sep="\t", nrows=0))
        inventory[path.name] = entry
    frames = {}
    result = {"files": inventory, "periods": {}}
    for part in ["train_val_test", "evaluation"]:
        path = raw / f"{part}_online_dataset.csv"
        data = pd.read_csv(path, sep="\t", parse_dates=["DATETIME"]).set_index("DATETIME")
        assert data.index.is_monotonic_increasing and data.index.is_unique
        intervals = data.index.to_series().diff().dropna()
        gaps = intervals[intervals != pd.Timedelta(minutes=15)]
        full = data.groupby(data.index.normalize()).filter(lambda group: len(group) == 96)
        use = full.loc["2021-05-06":"2021-11-09"] if part == "train_val_test" else full
        cols = [f"Q_air_{i}" for i in range(1, 6)]
        features = use[["Temp", "Q_inf"]].assign(air=use[cols].sum(axis=1, skipna=False))
        assert features.notna().all().all()
        frames[part] = features
        days = features.index.normalize()
        result["periods"][part] = dict(raw_start=str(data.index.min()), raw_end=str(data.index.max()),
            gaps_minutes={str(k): v.total_seconds()/60 for k,v in gaps.items()},
            all_full_days=full.index.normalize().nunique(), input_days=days.nunique(),
            first_day=str(days.min().date()), last_day=str(days.max().date()),
            min=features.min().to_dict(), max=features.max().to_dict(),
            mean_recorded_air_95=float(features.air.groupby(days).apply(lambda x: x.iloc[:95].sum()/4).mean()))
    original, later = frames.values()
    result["evaluation_fraction_outside_training_range"] = ((later < original.min()) | (later > original.max())).mean().to_dict()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    run(args.raw, args.output)
