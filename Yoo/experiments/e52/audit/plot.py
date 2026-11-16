"""고정 정책 분석 그림. 정책 seed의 대응 차이와 시간별 평균을 그린다."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main(a):
    data=json.loads((a.out/'analysis.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,title in zip(axes,['e50_actor','e52_fcsrl','e52_ppo'],['e50: Actor correction','e52: FCSRL pessimism','e52: PPO pessimism']):
        for p in data[key]['paired']:
            x=p['air_pct'];y=p['difference']['nh4_cost']
            ax.scatter(x,y,s=50)
            ax.annotate(f"seed {p['seed']}",(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.axhline(0,color='grey',linewidth=.8);ax.axvline(0,color='grey',linewidth=.8)
        ax.set_title(title);ax.set_xlabel('Air volume change (%)');ax.set_ylabel('NH4 cost change');ax.grid(alpha=.15);ax.margins(x=.22,y=.18)
    fig.tight_layout();fig.savefig(a.out/'paired_changes.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for j,m in enumerate(['fcsrl','ppo']):
        for k in [0,10]:
            air=[];nh4=[]
            for s in range(3):
                with np.load(a.root/'e52'/f'{m}_k{k}_s{s}'/'rollout.npz') as z:
                    air.append(z['air'].mean(0));nh4.append((z['states'][:,:,0]>4).mean(0)*100)
            for ax,values in [(axes[0,j],air),(axes[1,j],nh4)]:
                ax.plot((np.arange(95)+1)*.25,np.mean(values,0),label=f'kappa {k}')
                ax.grid(alpha=.15)
        axes[0,j].set_title(m.upper());axes[0,j].set_ylabel('Mean air rate (m3/h)');axes[0,j].legend()
        axes[1,j].set_ylabel('NH4 violations (%)');axes[1,j].set_xlabel('Hours after episode start')
    fig.suptitle('e52 validation: all dates, members and seeds; descriptive averages')
    fig.tight_layout();fig.savefig(a.out/'e52_time_profiles.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,default=HERE)
    main(p.parse_args())
