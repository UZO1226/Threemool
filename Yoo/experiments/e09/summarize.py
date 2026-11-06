"""e09 진단과 모델 대조 결과를 표 및 그림으로 집계한다."""
import argparse
import json
from pathlib import Path

import numpy as np


def main(a):
    audit = json.loads(a.audit.read_text())
    trial = json.loads(a.trial.read_text())
    assert 'completed_at' in trial and all(trial['checks'].values())
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in trial.items() if k not in ('evaluation', 'training')}
    summary['paired_training'] = trial['training']
    summary['evaluation'] = []
    keys = ('nh4_mae', 'nh4_bias', 'do_mae', 'predicted_cost', 'actual_cost', 'daily_cost_mae')
    for mode, rows in trial['evaluation'].items():
        groups = dict.fromkeys((r['split'], r['reset_steps'], r['group']) for r in rows)
        for split, h, group in groups:
            selected = [r for r in rows if (r['split'], r['reset_steps'], r['group']) == (split, h, group)]
            item = {'mode': mode, 'split': split, 'reset_steps': h, 'group': group,
                    'members': len(selected), 'days': selected[0]['days']}
            for key in keys:
                item[key] = {'mean': float(np.mean([r[key] for r in selected])),
                             'sd': float(np.std([r[key] for r in selected], ddof=0))}
            summary['evaluation'].append(item)
    (a.out / 'model_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (a.out / 'audit_summary.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    labels = {'original': '기존 모델', 'one_step_joint': '한 스텝 결합 추가 학습', 'multi_step12': '12스텝 결합 추가 학습'}
    md = ['# Yoo/e09 모델 보정 대조 결과', '',
          f"실행 커밋 `{trial['commit']}`. 종료 `{trial['completed_at']}` (UTC).", '',
          '원본 모델과 같은 구조의 구성원 5개를 각각 300회 추가 갱신했다. 값은 구성원 평균 ± 표준편차(ddof=0)이며 독립 학습 seed 편차가 아니다. 한 스텝과 12스텝 군은 같은 창과 샘플 순서를 사용했다. 자세한 [설정과 한계](README.md)를 따른다.', '',
          '비용은 NH4 > 4 및 DO > 2인 5분 전이를 하루 287개에 걸쳐 합산한다. 이 표는 정책 성능이 아니라 실제 공기량 재생의 정확도이며, 예측 비용을 실측에 가깝게 맞추는 것이 목적이다.', '']
    for split, group, title in [('val', 'all', '검증 전체 43일'), ('val', 'cold', '검증 저온 15일'),
                                 ('val', 'in_range', '검증 수온 범위 내 28일'), ('train', 'all', '학습 전체 330일')]:
        md += [f'## {title}', '', '| 방법 | 재시작 간격 (스텝) | NH4 MAE | DO MAE | 예측 비용 | 실측 비용 | 날짜별 비용 MAE |',
               '| --- | --- | --- | --- | --- | --- | --- |']
        for row in summary['evaluation']:
            if row['split'] != split or row['group'] != group:
                continue
            values = [f"{row[k]['mean']:.3f} ± {row[k]['sd']:.3f}" for k in ('nh4_mae','do_mae','predicted_cost')]
            md.append(f"| {labels[row['mode']]} | {row['reset_steps']} | " + ' | '.join(values)
                      + f" | {row['actual_cost']['mean']:.3f} | {row['daily_cost_mae']['mean']:.3f} ± {row['daily_cost_mae']['sd']:.3f} |")
        md += ['']
    md += ['## 검사', '', '- 기존 numpy 및 PyTorch의 12스텝 예측: 구성원 5개 모두 일치.',
           '- 두 학습군의 날짜와 창 샘플 해시: 모든 구성원 쌍에서 일치.',
           '- 모든 loss와 gradient 유한. 저장 모델 numpy 추론과 학습망 출력 일치.',
           '- 같은 날짜 안의 연속 5분 자료만 사용. val로 gradient 또는 checkpoint 선택을 하지 않았고 test 미사용.',
           '', '![검증 예측 길이별 오차](figures/rollout_error.png)', '']
    (a.out / 'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in labels:
        rows = [r for r in summary['evaluation'] if r['mode'] == mode and r['split'] == 'val' and r['group'] == 'all']
        for axis, key in zip(axes, ('nh4_mae', 'predicted_cost')):
            axis.errorbar([r['reset_steps'] for r in rows], [r[key]['mean'] for r in rows],
                          yerr=[r[key]['sd'] for r in rows], marker='o', capsize=3, label=mode)
    axes[1].axhline(rows[0]['actual_cost']['mean'], color='black', linestyle=':', label='Observed cost')
    for axis, title in zip(axes, ('NH4 MAE', 'Daily violation cost')):
        axis.set_xscale('log')
        axis.set_xlabel('Steps between observed-state resets')
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Validation replay: member mean and SD, 43 days')
    fig.tight_layout()
    (a.out / 'figures').mkdir(exist_ok=True)
    fig.savefig(a.out / 'figures/rollout_error.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--trial', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    main(p.parse_args())
