"""Plot all model candidates against the preregistered safety tolerances."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--cooperative',type=Path,required=True)
p.add_argument('--contractive',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
c=json.loads(a.cooperative.read_text());r=json.loads(a.contractive.read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
groups=[('Old delta',{'old':c['validation']['old']},'#64748b'),
        ('Previous direct',{'direct7':c['validation']['direct7']},'#2563eb'),
        ('Cooperative',{k:v for k,v in c['validation'].items() if k not in ['old','direct7','persistence']},'#d97706'),
        ('Contractive',{k:v for k,v in r['validation'].items() if k!='old'},'#059669')]
for label,rows,color in groups:
    for i,(name,v) in enumerate(rows.items()):
        axes[0].scatter(v['mae']['nh4'],v['nh4_violation']['absolute_error_pp'],color=color,
            marker='D' if 'cal' in name else 'o',label=label if i==0 else None,s=45)
        axes[1].scatter(v['normalized_mae'],v['do_violation']['absolute_error_pp'],color=color,
            marker='D' if 'cal' in name else 'o',s=45)
axes[0].axhline(5,color='black',ls=':')
axes[0].axvline(.9*c['validation']['old']['mae']['nh4'],color='black',ls=':')
axes[0].set(xlabel='NH4 recursive MAE (mg/L)',ylabel='NH4 violation frequency error (percentage points)',title='NH4: lower left is better')
axes[0].legend(fontsize=8)
axes[1].axhline(c['validation']['old']['do_violation']['absolute_error_pp'],color='black',ls=':')
axes[1].axvline(.9*c['validation']['old']['normalized_mae'],color='black',ls=':')
axes[1].set(xlabel='Normalized recursive MAE',ylabel='DO violation frequency error (percentage points)',title='DO calibration versus prediction error')
for ax in axes:ax.grid(alpha=.2)
a.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out,dpi=170)
