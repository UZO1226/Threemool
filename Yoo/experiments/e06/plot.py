"""고정 검증의 비용, CVaR와 공기량을 seed별로 표시한다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('ppolag', 'ppolag_tail', 'ppolag_fcsrl', 'ppolag_fcsrl_tail')
LABELS = ('PPO', 'PPO + tail', 'FCSRL + PPO', 'FCSRL + PPO + tail')
COLORS = ('#4379ac', '#67adcf', '#bf623b', '#dc9d47')


def plot(root, output):
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'complete'
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for row, seed in enumerate((0, 1)):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            log = json.loads((root / 'runs' / method / f'seed_{seed}' / 'train_log.json').read_text())
            points = [p for p in log if 'val' in p]
            x = [p['steps'] / 1000 for p in points]
            for col, metric in enumerate(('cost', 'cost_cvar', 'air_m3')):
                y = [p['val'][metric] / (1000 if metric == 'air_m3' else 1) for p in points]
                axes[row, col].plot(x, y, marker='o', markersize=4, color=color,
                                    linestyle='--' if method.endswith('_tail') else '-', label=label)
        axes[row, 0].axhline(15, color='gray', linestyle=':', label='Mean cost limit')
        for col, title in enumerate(('Mean daily cost', 'Upper 20% cost CVaR', 'Air (1,000 Nm3/day)')):
            axes[row, col].set_title(f'{title}, seed {seed}')
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xlabel('Training transitions (thousands)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    fig.suptitle('e06: fixed validation (43 days x 5 surrogate members), two exploratory seeds')
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'validation.png', dpi=180)
    fig.savefig(output / 'validation.svg')
    svg = output / 'validation.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.runs, args.out)
