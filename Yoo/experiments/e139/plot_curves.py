"""One page per period, showing every seed's measured air and water operating points."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pandas as pd

CONDITIONS = {
    "base": ("FCSRL", "#323841", "o"),
    "actor": ("Actor correction", "#0072b2", "s"),
    "base_risk": ("Risk selector", "#d55e00", "^"),
    "combined": ("Actor + selector", "#007a59", "D"),
}
DATES = {108: "22 Aug to 30 Sep", 148: "1 Oct to 9 Nov", 188: "10 Nov to 19 Dec (prior result)"}


def draw(points, path):
    df = pd.read_csv(points)
    with PdfPages(path) as pdf:
        for n, group in df.groupby("prefix"):
            fig, axes = plt.subplots(5,3,figsize=(10,11.6),sharex="row")
            fig.subplots_adjust(left=.09,right=.985,bottom=.07,top=.89,hspace=.48,wspace=.29)
            fig.suptitle(f"DARROW 2021   |   {DATES[n]}",x=.09,y=.975,ha="left",fontsize=16)
            legend = [Line2D([],[],color=color,marker=marker,label=label,markersize=4)
                      for label,color,marker in CONDITIONS.values()]
            fig.legend(handles=legend,loc="upper left",bbox_to_anchor=(.08,.95),ncol=4,frameon=False)
            for row,(seed,s) in enumerate(group.groupby("seed")):
                for col,(key,label) in enumerate([("nh4_min","NH4 violations (min/day)"),
                                                ("nh4_excess","NH4 excess (mg N h/L/day)"),
                                                ("do_min","DO violations (min/day)")]):
                    ax = axes[row,col]
                    for arm,(_,color,marker) in CONDITIONS.items():
                        data = s[s.arm==arm].sort_values("air_m3")
                        ax.plot(data.air_m3/1000,data[key],color=color,marker=marker,ms=3,lw=1)
                    if row==0:
                        ax.set_title(label,fontsize=10)
                    if col==0:
                        ax.set_ylabel(f"Seed {seed}",fontsize=10)
                    if row==4:
                        ax.set_xlabel("Daily air (1,000 Nm³)")
                    ax.spines[["top","right"]].set_visible(False)
                    ax.grid(axis="y",alpha=.18)
                    ax.tick_params(labelsize=8)
            pdf.savefig(fig)
            plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points",type=Path)
    parser.add_argument("pdf",type=Path)
    args=parser.parse_args()
    draw(args.points,args.pdf)
