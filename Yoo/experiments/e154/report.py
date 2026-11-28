"""Aggregate e154 control outcomes paired by day, policy seed and simulator member."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["air_m3","nh4_min","excess","do_min"]
ARMS = ("native","ungated","base","no3","neutral","refit")


def improves(frame):
    return (frame.air_m3 < -.05) & frame[METRICS[1:]].le(1e-8).all(axis=1)


def run(paths, out):
    summary, seed_rows = [], []
    for n in (108,148,188):
        tables = []
        for seed in range(10,15):
            table = pd.read_csv(paths/f"p{n}_s{seed}.csv")
            assert len(table) == 1200 and set(table.arm) == set(ARMS)
            assert table.prefix.eq(n).all() and table.seed.eq(seed).all()
            tables.append(table)
        data = pd.concat(tables,ignore_index=True)
        arms = {a:g.set_index(["seed","day","member"]).sort_index() for a,g in data.groupby("arm")}
        for a,g in arms.items():
            assert len(g) == 1000 and g.index.is_unique and g.index.equals(arms["native"].index)
            assert g.index.get_level_values("day").nunique() == 40
            assert g.index.get_level_values("member").unique().sort_values().tolist() == list(range(5))
        for a in ARMS[1:]:
            for b in (("native","ungated","base","no3","neutral") if a=="refit" else ("native",)):
                arm, comp = arms[a], arms[b]
                delta = arm[METRICS]-comp[METRICS]
                same = arm["first"]==comp["first"]
                np.testing.assert_array_equal(delta.loc[same],np.zeros((same.sum(),4)))
                means = delta.groupby(level="seed").mean()
                joint = improves(means)
                for s,row in means.iterrows():
                    seed_rows.append(dict(prefix=n,arm=a,control=b,seed=s,passed=joint.loc[s],**row.to_dict()))
                avg = delta.mean().to_frame().T
                summary.append(dict(prefix=n,arm=a,control=b,paths=len(arm),changed=(arm["first"]>=0).sum(),
                                    different_times=(~same).sum(),days_differ=arm.loc[~same].index.get_level_values("day").nunique(),
                                    **avg.iloc[0].to_dict(),seeds_pass=joint.sum(),mean_pass=improves(avg).iloc[0]))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary).to_csv(out/"control_summary.csv",index=False,float_format="%.10g")
    pd.DataFrame(seed_rows).to_csv(out/"control_seeds.csv",index=False,float_format="%.10g")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.paths,a.out)
