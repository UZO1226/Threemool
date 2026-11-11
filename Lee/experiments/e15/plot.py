"""Generate auditable figures from saved summaries and training logs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--summary',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
summary=json.loads(a.summary.read_text())
audit=json.loads(a.audit.read_text())
methods=['ppolag','ppolag_fcsrl','ppolag_fcsrl_curriculum']
labels=['PPO-Lag','FCSRL + PPO-Lag','FCSRL + risk sampling']
colors=['#64748b','#2563eb','#d97706']
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for method,label,color in zip(methods,labels,colors):
    rows=summary['per_seed'][method]
    x=[r['val_final']['air_m3']/1000 for r in rows]
    y=[r['val_final']['cost'] for r in rows]
    axes[0].scatter(x,y,color=color,alpha=.5,s=25)
    axes[0].errorbar(np.mean(x),np.mean(y),xerr=np.std(x),yerr=np.std(y),color=color,
        marker='D',capsize=3,label=label)
    for seed in range(5):
        log=json.loads((a.runs/method/f'seed_{seed}'/'train_log.json').read_text())
        val=[r for r in log if 'val' in r]
        axes[1].plot([r['steps']/1000 for r in val],[r['val']['cost'] for r in val],color=color,alpha=.25)
axes[0].set(xlabel='Air volume (1000 Nm3 / 23.75 h)',ylabel='Constraint cost',title='Final policies, five seeds (mean +/- SD)')
axes[0].legend(fontsize=8)
axes[1].set(xlabel='Training transitions (1000)',ylabel='Validation constraint cost',title='Every seed, all validation checkpoints')
axes[1].axhline(11,ls=':',color='black',label='Training cost limit')
for ax in axes: ax.grid(alpha=.2)
fig.savefig(a.out/'controller_comparison.png',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
names=['old','persistence','direct_7','direct_15']
pretty=['Old delta GBM','Persistence','Direct 7','Direct 15']
axes[0].bar(pretty,[audit['scores'][n]['rollout_mae']['nh4'] for n in names],color=['#64748b','#94a3b8','#2563eb','#60a5fa'])
axes[0].set(ylabel='NH4 recursive MAE (mg N/L)',title='Environment validation, 95-step rollouts')
axes[0].tick_params(axis='x',rotation=15)
pos=np.arange(3)
for offset,key,label in [(-.18,'nh4_viol_pct','NH4 above limit'),(.18,'do_viol_pct','DO above limit')]:
    axes[1].bar(pos+offset,[summary['summary']['final'][m][key]['mean'] for m in methods],.36,label=label)
axes[1].set(xticks=pos,xticklabels=['PPO-Lag','FCSRL','Risk sampling'],ylabel='Violation steps (%)',title='Final policy violations, separate constraints')
axes[1].legend(fontsize=8)
fig.savefig(a.out/'environment_and_violations.png',dpi=170)
plt.close(fig)
