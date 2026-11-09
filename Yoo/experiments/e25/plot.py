"""마지막 정책 seed와 같은 공기량 비용 차이를 그림으로 남긴다."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
s = json.loads(a.summary.read_text())
font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for i, tag in enumerate(('mlp', 'gbm')):
    ax = axes[0, i]
    for method, label, color in [('ppo', 'PPO', '#777777'), ('fcsrl', '기본 FCSRL', '#0072B2'), ('fcsrl_preserve', '표현 반영 보정', '#D55E00')]:
        r = s['raw'][tag][method]
        ax.scatter(r['air_m3']['values'], r['cost']['values'], color=color, alpha=.65, label=label)
        ax.scatter(r['air_m3']['mean'], r['cost']['mean'], marker='X', s=120, color=color, edgecolor='black')
    ax.axhline(15, color='black', ls=':', lw=1, label='비용 한도 15')
    ax.set(title=tag.upper() + ' 최종 정책 (X는 평균)', xlabel='평균 공기량 (Nm³/일)', ylabel='하루 위반 비용')
    ax.legend(fontsize=9)
    ax = axes[1, i]
    r = s['controls'][tag]['paired_cost_difference']
    colors = ['#0072B2' if x < 0 else '#D55E00' for x in r['values']]
    ax.bar(range(5), r['values'], color=colors, alpha=.8)
    ax.axhline(0, color='black', lw=1)
    lo, hi = r['ci95']
    ax.set(title=f"같은 공기량 비용 차이, 평균 {r['mean']:.2f}\n95% CI [{lo:.2f}, {hi:.2f}]", xlabel='seed', ylabel='확장 비용 빼기 기본 비용 (낮을수록 좋음)', xticks=range(5))
fig.suptitle('Kaggle 대리 환경 e25, seed 5개, 실제 공정 안전성 검증 아님', fontsize=14)
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=180)
plt.close(fig)
