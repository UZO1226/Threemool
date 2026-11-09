"""Kaggle 모델 내 공기량과 비용, 실제 기록의 차이를 함께 표시한다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
s = json.loads(a.summary.read_text())
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
methods = [('ppo', 'PPO'), ('ppo_anchor', 'PPO + behavior penalty'),
           ('fcsrl', 'FCSRL + PPO'), ('fcsrl_anchor', 'FCSRL + PPO + behavior penalty')]
for ax, tag in zip(axs, ('mlp', 'gbm')):
    for (method, label), color in zip(methods, ('C0', 'C1', 'C2', 'C3')):
        rs = [r['final'][tag] for r in s['runs'] if r['method'] == method]
        x, y = [r['air_m3'] for r in rs], [r['cost'] for r in rs]
        ax.scatter(x, y, color=color, alpha=.45, s=25)
        ax.errorbar(np.mean(x), np.mean(y), xerr=np.std(x, ddof=1), yerr=np.std(y, ddof=1),
                    fmt='o', color=color, capsize=3, label=label)
    for r, marker, label in [(s['behavior_policy'][tag], 's', 'Learned behavior policy'),
                             (s['historical_replay'][tag], '^', 'Historical air in surrogate'),
                             (s['historical_measured'], '*', 'Historical measurements')]:
        ax.scatter(r['air_m3'], r['cost'], marker=marker, c='black', s=70, label=label)
    ax.axhline(15, ls=':', c='red', label='Cost limit 15')
    ax.set_title(f'{tag.upper()}: validation, 43 days x 5 members')
    ax.set_xlabel('Air volume (Nm3/day)')
    ax.grid(alpha=.2)
axs[0].set_ylabel('Mean daily violation cost')
axs[1].legend(fontsize=7, loc='upper left')
fig.suptitle('Kaggle Avedore: final policies, 2 seeds (mean +/- sample SD)')
fig.tight_layout()
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=160)
