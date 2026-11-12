"""Display the fixed-algorithm baseline cross comparison without claiming dominance."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
records=[]
for name in ('old_baseline_validation.json','new_fcsrl_validation.json'):
    source=json.loads((ROOT/'Lee/experiments/e42'/name).read_text());assert not source['test_evaluated']
    records.extend(source['records'])
groups={};summary={}
for evaluation in ('old','new'):
    summary[evaluation]={}
    for training in ('old','new'):
        selected=sorted([r for r in records if r['evaluation_environment']==evaluation and r['training_environment']==training],key=lambda r:r['policy_seed'])
        assert [r['policy_seed'] for r in selected]==list(range(5))
        groups[evaluation,training]=selected;summary[evaluation][training]={}
        for key in ('air_m3','cost','nh4_viol_pct','do_viol_pct'):
            values=np.array([r['metrics'][key] for r in selected])
            summary[evaluation][training][key]={'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'policy_seed_values':values.tolist()}
fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
colors={'old':'#2563eb','new':'#dc2626'}
for row,evaluation in enumerate(('old','new')):
    for col,(key,title,scale) in enumerate([('cost','Cost (counts/day)',1),('nh4_viol_pct','NH4 violation (%)',1),
            ('do_viol_pct','DO violation (%)',1),('air_m3','Air (1000 m3 / 23.75 h)',1000)]):
        ax=axes[row,col]
        for x,training in enumerate(('old','new')):
            values=np.array(summary[evaluation][training][key]['policy_seed_values'])/scale
            ax.scatter(x+np.linspace(-.08,.08,5),values,color=colors[training],s=25,alpha=.8)
            ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt='s',color='black',capsize=5,markersize=5)
        if key=='cost':ax.axhline(12,color='grey',ls='--',lw=1)
        ax.set(xticks=[0,1],xticklabels=['Train original','Train e40'],title=title,ylim=(0,None))
        ax.grid(axis='y',alpha=.2)
        if col==0:ax.set_ylabel('Evaluate original' if evaluation=='old' else 'Evaluate e40')
fig.suptitle('Final FCSRL policies: training-model sensitivity, five policy seeds')
fig.supxlabel('Each row holds the evaluation simulator fixed. Dots: policy seeds. Black: mean and sample SD, not confidence intervals.\nNative actions, air not matched. Cross-simulator scores are separate sensitivity conditions, not real-plant validation.',fontsize=9)
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=160)
a.summary.write_text(json.dumps({'metrics_by_evaluation_then_training':summary,'test_evaluated':False,
    'scope':'Fixed FCSRL algorithm, native final policies. No extension comparison or air-matched dominance claim.'},indent=2))
print('Cross-model plot and compact summary saved',flush=True)
