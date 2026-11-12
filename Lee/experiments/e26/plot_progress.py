"""Show safety-frequency failure separately from improving forecast error."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()

def read(experiment):
    return json.loads((ROOT/f'Lee/experiments/{experiment}/summary.json').read_text())

old=read('e16')['validation'];rows=[('Persistence',old['persistence']),('Original',old['old']),('e24',read('e24')['validation'])]
for experiment in [f'e{k}' for k in range(26,41)]:
    if not (ROOT/f'Lee/experiments/{experiment}/summary.json').exists():continue
    source=read(experiment);value=source['validation']
    if 'normalized_mae' not in value:value=value[source['selected']]
    rows.append((experiment,value))

fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout='constrained')
for ax,key,title,tolerance in [(axes[0],'nh4_violation','NH4 violation frequency',5),
        (axes[1],'do_violation','DO violation frequency',min(1,old['old']['do_violation']['absolute_error_pp']))]:
    actual=old['old'][key]['actual'];x=np.arange(len(rows));means=np.array([value[key]['pred'] for _,value in rows])
    low=np.array([min(value[key]['member_pred']) for _,value in rows]);high=np.array([max(value[key]['member_pred']) for _,value in rows])
    ax.axhspan(max(0,actual-tolerance),actual+tolerance,color='#dcfce7',zorder=0)
    ax.axhline(actual,color='black',ls='--',lw=1,label=f'Observed {actual:.2f}%')
    ax.errorbar(x,means,yerr=np.stack([means-low,high-means]),fmt='o',capsize=4,color='#2563eb',markersize=5)
    for i,mean in enumerate(means):ax.annotate(f'{mean:.1f}',(i,mean),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=[label for label,_ in rows],ylabel='Event steps (%)',title=title)
    ax.tick_params(axis='x',rotation=45);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.supxlabel('Exploratory reused development40, recorded-air replay. Bands: ensemble frequency criterion only.\nWhiskers: member range, not confidence intervals. Adoption also requires forecast-error and member checks.',fontsize=9)
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=160)
