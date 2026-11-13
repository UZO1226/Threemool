"""Plot complete five-seed matched-air effects without new result selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs=4,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
reports=[json.loads(path.read_text()) for path in a.reports]
names=['Split heads','Capacity control','Severity label','Cold-day sampling']
expected=['ppolag_fcsrl_split','ppolag_fcsrl_capacity','ppolag_fcsrl_severity','ppolag_fcsrl_cold']
keys=['cost','nh4_viol_pct','do_viol_pct']
for i,report in enumerate(reports):
    assert report['candidate_method']==expected[i] and report['policy_seeds']==list(range(5))
    assert not report['test_evaluated'] and report['checks']['all_five_seeds_air_matched']
    assert report['freeze_canonical_sha256']==reports[0]['freeze_canonical_sha256']
    for key in keys:
        np.testing.assert_array_equal(report['metrics']['candidate_matched'][key]['baseline_values'],
            reports[0]['metrics']['candidate_matched'][key]['baseline_values'])
fig,axes=plt.subplots(1,3,figsize=(12,4.7),sharey=True)
labels=['Total cost difference','NH4 difference (percentage points)','DO difference (percentage points)']
for ax,key,label in zip(axes,keys,labels):
    ax.axvline(0,color='.45',linestyle=':',linewidth=1.2)
    for i,report in enumerate(reports):
        row=report['metrics']['candidate_matched'][key]
        mean=row['paired_mean_difference'];lo,hi=row['paired_seed_bootstrap95_ci']
        ax.scatter(row['paired_difference'],i+np.linspace(-.10,.10,5),color='.65',s=21,zorder=2)
        ax.plot([lo,hi],[i,i],color='#176f9b',linewidth=2.4,zorder=3)
        ax.scatter([mean],[i],marker='D',color='#073d59',s=37,zorder=4)
    ax.set_xlabel(label);ax.grid(axis='y',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(range(4),names);axes[0].invert_yaxis()
fig.suptitle('Candidate minus FCSRL at matched air',fontsize=15,y=.98)
fig.text(.5,.055,'Grey dots: five policy seeds. Diamond and line: mean and paired 95% seed-bootstrap interval.',ha='center',fontsize=10)
fig.text(.5,.015,'Reused validation, frozen surrogate. All four candidates fail the complete improvement criterion.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.10,1,.94));a.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out,dpi=180);plt.close(fig)
print(a.out,flush=True)
