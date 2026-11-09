"""정책 seed별 공기량과 비용, 같은 공기량 대비 차이를 그린다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'fcsrl': 'Baseline', 'fcsrl_actor': 'Actor only', 'fcsrl_critic': 'Critics only',
          'fcsrl_preserve': 'Actor + critics (e25)', 'fcsrl_extra_ppo': 'Extra PPO'}
COLORS = dict(zip(LABELS, ['#666666', '#2563eb', '#d97706', '#059669', '#9333ea']))


def main(a):
    data = json.loads(a.summary.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for col, tag in enumerate(('mlp', 'gbm')):
        ax = axes[0, col]
        for method, r in data['raw'][tag].items():
            x = np.asarray(r['air_m3']['values']) / 1000
            y = r['cost']['values']
            ax.scatter(x, y, c=COLORS[method], alpha=.55, s=35)
            ax.scatter(x.mean(), np.mean(y), c=COLORS[method], s=100, marker='X',
                       edgecolor='black', linewidth=.5, label=LABELS[method])
        ax.axhline(15, color='red', linestyle=':', linewidth=1, label='Mean cost limit')
        ax.set(title=f'{tag.upper()}: final policies, 5 seeds', xlabel='Daily air (1000 Nm3)', ylabel='Daily cost')
        ax.grid(alpha=.2)
        if col == 0:
            ax.legend(fontsize=8)
        ax = axes[1, col]
        for row, method in enumerate(('fcsrl', 'fcsrl_actor', 'fcsrl_critic', 'fcsrl_extra_ppo')):
            r = data['comparisons'][tag][method]
            ax.scatter(r['values'], [row] * 5, c=COLORS[method], alpha=.5, s=30)
            ax.errorbar(r['mean'], row, xerr=[[r['mean'] - r['ci95'][0]], [r['ci95'][1] - r['mean']]],
                        fmt='D', color=COLORS[method], capsize=4)
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
        ax.set_yticks(range(4), [LABELS[m] for m in ('fcsrl', 'fcsrl_actor', 'fcsrl_critic', 'fcsrl_extra_ppo')])
        ax.invert_yaxis()
        ax.set(title=f'{tag.upper()}: matched-air contrasts', xlabel='Cost: full preservation minus comparator\nDots: paired seeds; bars: unadjusted 95% t CI')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('e29: preservation ablations on reused Kaggle validation data\nSurrogate results, not measured plant performance', fontsize=13)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
