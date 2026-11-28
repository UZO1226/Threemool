"""Plot the prespecified closest discordant pair on common clock and axes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(source, dest):
    df = pd.read_csv(source/'example.csv')
    pair = pd.read_csv(source/'pairs.csv').query('opposite and gap <= 1000').sort_values(
        ['gap','day','member','seed_lo','seed_hi']).iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey='row')
    for col, (seed, prefix) in enumerate([(pair.seed_lo, 'lo'), (pair.seed_hi, 'hi')]):
        group = df[df.seed == seed].sort_values('step')
        x = np.arange(96)/4
        for arm, color in [('uniform','#555d67'),('weighted','#087e83')]:
            axes[0,col].stairs(group[arm+'_air'], x, baseline=None, color=color, label=arm.title())
            nh = np.r_[group[arm+'_pre'].iloc[0], group[arm+'_next']]
            axes[1,col].plot(x, nh, color=color, label=arm.title())
        start, length = pair[prefix+'_start_min']/60, pair[prefix+'_duration']/60
        for ax in axes[:,col]:
            ax.axvspan(start, start+length, color='#d9a637', alpha=.18, zorder=0)
            ax.spines[['top','right']].set_visible(False)
        axes[1,col].axhline(4, color='#a64d38', ls=':', lw=1.5)
        axes[0,col].set_title(f'Seed {int(seed)}    |    NH4 violation steps {int(pair["nh_"+prefix]):+d}')
        axes[1,col].set_xlabel('Hour of simulated day')
        axes[1,col].set_xticks([0,4,8,12,16,20,24])
    axes[0,0].set_ylabel('Air flow (m³/h)')
    axes[1,0].set_ylabel('NH4 (mg N/L)')
    fig.suptitle(f'{pair.day}    |    Surrogate member {int(pair.member)}')
    axes[0,0].legend(loc='upper left', frameon=False)
    fig.text(.5,.015,'Shading marks the longest spell at least 250 m³/h below the same-seed uniform policy.',ha='center')
    fig.tight_layout(rect=(0,.045,1,.95))
    fig.savefig(dest, dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('dest',type=Path)
    args=parser.parse_args()
    plot(args.source,args.dest)
