"""Elapsed time until the first offset and the final return to nonnegative extra air."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

source, output = map(Path, sys.argv[1:])
cases = pd.read_csv(source / "cases.csv")
losses = cases[(cases["mode"]=="matched") & (cases.group=="large")].sort_values(["first","lasting"])
fig, ax = plt.subplots(figsize=(7.5,4.4), layout="constrained")
y = np.arange(len(losses))
a,b = losses["first"]/60, losses.lasting/60
ax.hlines(y,a,b,color="#7a8d9b",linewidth=2)
ax.scatter(a,y,marker="o",facecolors="white",edgecolors="#26638b",s=60,label="First offset",zorder=3)
ax.scatter(b,y,marker="|",color="#ab4823",s=135,linewidths=2.4,label="Offset lasting to day end",zorder=4)
ax.set_yticks(y,[f"{r.day[5:]}   S{r.seed} / M{r.member}" for r in losses.itertuples()],fontsize=8)
ax.invert_yaxis()
ax.set(xlim=(0,max(b)+1),xlabel="Hours since the 15-minute reduction",ylabel="Date, policy seed / model",
       title="When the initial air saving was offset in large-loss episodes")
ax.legend(frameon=False,loc="lower right",fontsize=8)
ax.grid(axis="x",color="#dddddd",linewidth=.6)
ax.spines[["top","right"]].set_visible(False)
fig.savefig(output,dpi=150)
