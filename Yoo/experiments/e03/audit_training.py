"""선택 모델과 마지막 epoch의 검증 지표 및 승수 기록을 모은다."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FLOAT = r'[-+\d.eE]+'
EPOCH = re.compile(r'Epoch #(\d+) =+\s*test_reward: (' + FLOAT + r')[^\n]*\n'
                   r'test_cost: (' + FLOAT + r')')
PROGRESS = re.compile(r'Epoch #(\d+):\s+100%[^\n]*?cost=(' + FLOAT + r'), lambda=(' + FLOAT + r')')


def history(run):
    ppo = run / 'train_log.json'
    if ppo.exists():
        return json.loads(ppo.read_text())
    files = list(run.glob('wandb/offline-run-*/files/output.log'))
    if len(files) != 1:
        raise ValueError(f'Expected one training log in {run}, got {files}')
    text = files[0].read_text()
    progress = {int(ep): (float(cost), float(lam)) for ep, cost, lam in PROGRESS.findall(text)}
    return [{'epoch': int(ep), 'test_reward': float(reward), 'test_cost': float(cost),
             'lambda': progress[int(ep)][1], 'last_training_cost': progress[int(ep)][0]}
            for ep, reward, cost in EPOCH.findall(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for root in a.runs:
        for p in sorted(root.glob('*/seed_*/metrics.json')):
            m = json.loads(p.read_text())
            tag = m.get('tag', m['method'])
            if tag == 'td3':
                continue
            h = history(p.parent)
            if [r['epoch'] for r in h] != list(range(1, 26)):
                raise ValueError(f'Incomplete epoch history: {p}')
            selected = h[m['best_epoch'] - 1]
            rows.append({'tag': tag, 'seed': m['seed'], 'best_epoch': m['best_epoch'],
                         'selected_val_cost': selected['test_cost'], 'selected_test_cost': m['test_best']['cost'],
                         'final_val_cost': h[-1]['test_cost'], 'final_test_cost': m['test_final']['cost'],
                         'final_lambda': h[-1]['lambda'], 'history': h})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'training_audit.json').write_text(json.dumps(rows, indent=2))
    md = ['# 학습 기록 점검', '',
          '검증 비용은 모델 선택에 사용한 무작위 10 에피소드의 평균이다. 평가 비용은 고정 300 에피소드 평균이다.',
          'TD3 계열의 마지막 lambda는 로그에 소수 둘째 자리로 표시된 값이다. PPO는 train_log.json의 값이다.', '',
          '| 방법 | seed | 선택 epoch | 선택 검증 비용 | 선택 평가 비용 | 마지막 평가 비용 | 마지막 lambda |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        md.append(f"| {r['tag']} | {r['seed']} | {r['best_epoch']} | {r['selected_val_cost']:.2f} | "
                  f"{r['selected_test_cost']:.2f} | {r['final_test_cost']:.2f} | {r['final_lambda']:.3f} |")
    (a.out / 'training_audit.md').write_text('\n'.join(md) + '\n')
    order = ['fcsrl', 'fcsrl_ucb', 'fcsrl_ml5', 'fcsrl_ucb_ml5', 'ppolag']
    fig, axs = plt.subplots(2, len(order), figsize=(14, 5), sharex=True)
    for j, tag in enumerate(order):
        group = [r for r in rows if r['tag'] == tag]
        for r in group:
            x = [v['epoch'] for v in r['history']]
            axs[0, j].plot(x, [v['lambda'] for v in r['history']], alpha=.6, lw=1)
            axs[1, j].plot(x, [v['test_cost'] for v in r['history']], alpha=.6, lw=1)
        axs[0, j].set_title(tag, fontsize=9)
        axs[1, j].axhline(15, color='black', ls='--', lw=.8)
        axs[1, j].set_xlabel('Validation round')
    axs[0, 0].set_ylabel('Multiplier (scales differ)')
    axs[1, 0].set_ylabel('Validation cost (10 episodes)')
    fig.tight_layout()
    (a.out / 'figures').mkdir(exist_ok=True)
    fig.savefig(a.out / 'figures/training_e03.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
