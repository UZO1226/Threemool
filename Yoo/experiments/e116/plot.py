"""Plot paired intervention effects at the 15-minute decision and observation times."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot(path, dest):
    f=pd.read_csv(path).query('pulse==0').copy()
    f['volume']=f.groupby('member').delta_air.cumsum()/4
    grouped=f.groupby('member')
    others=f[f.member!=3]
    avg=others.groupby('t').mean(numeric_only=True)
    fig,axes=plt.subplots(3,1,figsize=(9,7.4),sharex=True)
    for ax,col,clock,ylabel in zip(axes,['delta_nh4','delta_air','volume'],
                                    ['end_min','start_min','end_min'],
                                    ['NH4 difference\n(mg N/L)','Air flow difference\n(m³/h)','Cumulative air\ndifference (m³)']):
        for m,g in grouped:
            if m!=3:
                ax.plot(g[clock]/60,g[col],color='#9bafbf',lw=.85,alpha=.6)
        ax.plot(avg[clock]/60,avg[col],color='#2a657d',lw=2,label='Mean of members 0, 1, 2, 4')
        g=grouped.get_group(3)
        ax.plot(g[clock]/60,g[col],color='#b04623',lw=2,label='Member 3')
        ax.axhline(0,color='#555555',lw=.7)
        ax.axvspan(12,18,color='#a79765',alpha=.12)
        ax.set_ylabel(ylabel)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper left',frameon=False,ncol=2)
    axes[-1].set_xlabel('Hour of simulated day')
    axes[-1].set_xticks([0,3,6,9,12,15,18,21,24])
    fig.suptitle('6 December 2021   |   Midnight pulse minus no intervention')
    fig.tight_layout()
    fig.savefig(dest,dpi=155)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',type=Path)
    parser.add_argument('dest',type=Path)
    a=parser.parse_args()
    plot(a.paths,a.dest)
