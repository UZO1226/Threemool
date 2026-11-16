"""고정 여유 대조와 기록 행동 모델 진단을 그린다."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    diag=json.loads((HERE/'model_diagnostic.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for ax,split in zip(axes,['train','val']):
        z=diag['splits'][split]
        values=[z['observed_state_one_step']['actual_cost'],z['observed_state_one_step']['predicted_cost'],z['recursive_recorded_action']['predicted_cost']]
        ax.bar(['Observed','One-step','Recursive'],values,color=['#777777','#4098d7','#e69739'])
        ax.set_title(split);ax.set_ylabel('Mean NH4 violation count / 95 steps')
        ax.set_ylim(0,max(values)*1.2)
        for i,v in enumerate(values):ax.text(i,v+.3,f'{v:.2f}',ha='center')
    fig.suptitle('Recorded actions; original NH4 threshold; no safety margin')
    fig.tight_layout();fig.savefig(HERE/'model_replay_bias.png',dpi=170);plt.close(fig)
    p=HERE/'comparison.json'
    if not p.exists():return
    data=json.loads(p.read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for j,m in enumerate(['ppo','fcsrl']):
        z=data['comparisons'][m]
        for c,label in [('k0','No margin'),('k10','Adaptive margin'),('d045','Fixed margin')]:
            row=z['means'][c]
            axes[0,j].scatter(row['air_m3']/1000,row['nh4_cost'],s=55,label=label)
        axes[0,j].set_title(m.upper()+' (means of 3 seeds)')
        axes[0,j].set_xlabel('Air volume (1000 m3 / 23.75h)');axes[0,j].set_ylabel('NH4 violation count')
        axes[0,j].axhline(6,color='grey',ls=':',lw=1);axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.2)
        for seed,pair in enumerate(z['paired']):
            x=pair['nh4_cost'];y=pair['nh4_excess_mg_h_L']
            axes[1,j].scatter(x,y,s=55)
            axes[1,j].annotate(f'seed {seed}',(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
        axes[1,j].axhline(0,color='grey',lw=1);axes[1,j].axvline(0,color='grey',lw=1)
        axes[1,j].set_xlabel('NH4 count difference (adaptive - fixed)');axes[1,j].set_ylabel('NH4 excess integral difference (mg h/L)')
        axes[1,j].margins(x=.25,y=.25);axes[1,j].grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'margin_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
