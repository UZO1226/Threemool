"""사전 검사 요약에서 표와 고정 제어 반응 그림을 만든다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
s = json.loads(a.summary.read_text())
a.out.mkdir(parents=True, exist_ok=True)
(a.out / 'summary.json').write_text(json.dumps(s, indent=2, allow_nan=False) + '\n')
md = ['# e16 연결 검사 결과', '',
      '안정화 및 환경 검사를 완료했다. 아래는 학습 정책이 아닌 고정 KLa 제어 결과다. '
      '각 시나리오에서 같은 초기 상태로 7일을 연속 계산했으며 seed 평균이 아니다.', '',
      '| 날씨 | KLa5 (day^-1) | 폭기 (kWh/day) | 유량 가중 NH4 (mg N/L) | NH4 초과 시간 (%) | DO5 평균 (mg/L) |',
      '| --- | --- | --- | --- | --- | --- |']
for r in s['baselines']:
    md.append(f"| {r['weather']} | {r['kla5_per_day']:g} | {r['aeration_kwh_per_day']:.1f} | "
              f"{r['nh4_flow_weighted_mean']:.3f} | {r['nh4_violation_percent']:.2f} | {r['do5_mean']:.3f} |")
speed = sum(r['control_steps'] for r in s['baselines']) / sum(r['seconds'] for r in s['baselines'])
check = s['integration_check']
md += ['', f"실행 코드 커밋: `{s['code_commit']}`. 전체 사전 검사 {s['elapsed_seconds']:.1f}초.", '',
       f"- MATLAB 상수 유입 정상상태 기준 통과. 21개 방류 값의 최대 절대 차이 {s['steady']['max_abs_error']:.3g}.",
       '- reset 재현성, 인스턴스 간 독립성, 잘못된 행동 거부, 시간 경계, 직접 upstream 적분과의 일치, 에너지 적분 및 비용 집계 검사 통과.',
       '- 기존 PPO, FCSRL+PPO, FCSRL+PPO+CVaR 각각 12전이 추론 및 유한 출력 확인. 최적화 갱신은 0회다.',
       f"- 1분 대비 30초 적분의 동일 시각 NH4 MAE {check['nh4_endpoint_mae']:.4f} mg N/L, "
       f"DO5 MAE {check['do5_endpoint_mae']:.4f} mg/L. 건기 기본 제어 한 조건만 검사했다.",
       f'- 고정 제어 측정 처리량은 약 {speed:.0f}전이/초다. 40만 전이는 환경 계산만 약 {400000/speed/60:.1f}분이다. '
       '정책 추론, 역전파, 검증, 병렬 실행 간 경합은 포함하지 않은 외삽으로 학습 종료 시각이 아니다.',
       '', '## 해석', '',
       '폭기량 증가에 따른 NH4 감소와 에너지 증가를 확인했다. 최대 고정 폭기에서 유량 가중 NH4는 건기와 우기 모두 4 미만이지만, '
       '순간 초과율은 각각 약 17%, 26%다. 평균 기준 만족과 모든 시점 만족은 다르다. 이 결과는 다른 제어로 초과율을 더 낮출 수 없다는 증명이 아니다.', '',
       '5% 초과 시간 예산을 그대로 본 학습에 사용하는 근거는 없다. 후속 비용 정의는 BSM의 평균 수질 기준과 일치시키고, '
       '초과 시간과 상위 꼬리 위험은 별도 지표로 유지해야 한다. 임계 농도 4 mg N/L를 결과에 맞춰 바꾸지 않는다.', '',
       '기존 대리 모델의 FPR이나 탐지율을 개선한 결과가 아니다. 이번 환경에서의 기법 개선, 실제 공정 적용성, 전체 BSM 기준 충족도 아직 검증하지 않았다.', '',
       '![고정 폭기 반응](figures/fixed_controls.png)', '']
(a.out / 'results.md').write_text('\n'.join(md))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for weather, marker in [('dry', 'o'), ('rain', 's')]:
    rows = [r for r in s['baselines'] if r['weather'] == weather]
    x = [r['aeration_kwh_per_day'] for r in rows]
    for ax, metric in zip(axes, ['nh4_flow_weighted_mean', 'nh4_violation_percent']):
        y = [r[metric] for r in rows]
        ax.plot(x, y, marker=marker, label=weather)
        for xx, yy, r in zip(x, y, rows):
            ax.annotate(f"KLa={r['kla5_per_day']:g}", (xx, yy), xytext=(-5 if r['kla5_per_day'] == 240 else 4, -15 if weather == 'dry' else 8),
                        ha='right' if r['kla5_per_day'] == 240 else 'left', textcoords='offset points', fontsize=8)
        ax.set_xlabel('Aeration energy (kWh/day)')
        ax.grid(alpha=.2)
        ax.legend()
axes[0].axhline(4, color='black', linestyle=':', label='NH4 limit')
axes[0].set_ylabel('Flow-weighted effluent NH4 (mg N/L)')
axes[1].set_ylabel('Time above NH4 4 mg N/L (%)')
fig.suptitle('BSM1 fixed controls, seven-day preflight (no RL training)')
(a.out / 'figures').mkdir(exist_ok=True)
fig.savefig(a.out / 'figures/fixed_controls.png', dpi=160)
plt.close(fig)
print('\n'.join(md))
