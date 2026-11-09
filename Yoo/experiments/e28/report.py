"""e28 동일 시각 대조의 결과와 해석을 기록한다."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(source, out):
    r = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    avg = lambda rows, key: float(np.mean([v[key] for v in rows]))
    gates = []
    for mode in r['config']['modes']:
        for split in ('inner', 'val'):
            for seed in (0, 1):
                rows = [v for v in r['rows'] if v['mode'] == mode and v['split'] == split and v['seed'] == seed]
                base = next(v for v in rows if v['horizon'] == 'free')
                for v in rows:
                    assert (v['tp']+v['fn'], v['fp']+v['tn']) == (base['tp']+base['fn'], base['fp']+base['tn'])
                    if v['horizon'] != 'free':
                        gates.append(dict(mode=mode, split=split, seed=seed, horizon=v['horizon'],
                            recall_not_worse=v['recall'] >= base['recall'], fpr_lower=v['fpr'] < base['fpr'],
                            mae_lower=v['nh4_mae'] < base['nh4_mae']))
    r['fixed_target_checks'] = gates
    (out/'summary.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
    md = ['# e28 동일 목표 시각 대조 결과', '',
          f"실행 `{r['commit']}`, seed0,1 평균. 내부66일, 외부43일, 가중치 고정. 새 학습/test 사용 없음.", '',
          '아래 수치는 실제 시작 상태를 제공한 예측 진단이다. 하루 자율 제어 성능 개선으로 해석하지 않는다.', '',
          '## 동일 후반 목표13:05~23:55, 예측 길이 대조', '',
          '| 구간 | 모델 | 예측 전이 수 | NH4 recall | FPR | AUC | NH4 MAE |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for split in ('inner', 'val'):
        for mode in r['config']['modes']:
            for horizon in ('free', *r['config']['horizons']):
                rows = [v for v in r['rows'] if v['mode'] == mode and v['split'] == split and v['horizon'] == horizon]
                md.append(f"| {split} | {mode} | {horizon} | {100*avg(rows, 'recall'):.2f}% | {100*avg(rows, 'fpr'):.2f}% | {avg(rows, 'auc'):.4f} | {avg(rows, 'nh4_mae'):.3f} |")
    md += ['', 'free는01:00 실측 상태에서 시작한 원래 재생이다. 나머지는 각 목표의 h전이 전 실측 NH4/DO에서 시작한다. 모든 행은 동일 날짜/목표/정답/외생 경로를 공유한다.', '',
           '## 3시간 전 상태의 NH4/DO 분리 대조', '',
           '| 구간 | 모델 | 시작 정보 | NH4 recall | FPR | AUC | NH4 MAE |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for split in ('inner', 'val'):
        for mode in r['config']['modes']:
            for initial in ('none', 'nh4', 'do', 'both'):
                rows = [v for v in r['interventions'] if v['mode'] == mode and v['split'] == split and v['initial'] == initial]
                md.append(f"| {split} | {mode} | {initial} | {100*avg(rows, 'recall'):.2f}% | {100*avg(rows, 'fpr'):.2f}% | {avg(rows, 'auc'):.4f} | {avg(rows, 'nh4_mae'):.3f} |")
    md += ['', 'none은 두 상태 모두 기존 하루 예측, nh4/do는 해당 시작 상태만 실측, both는 두 시작 상태를 실측으로 바꾼다. 이후36전이에는 실측 수질을 넣지 않는다. DO의 실제 다음 값도 넣지 않는다.', '',
           '## day_mse 고정 예측 길이의 시간대 비교', '',
           '| 구간 | 예측 전이 수 | 목표 시간대 | recall | FPR | AUC | NH4 MAE |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for split in ('inner', 'val'):
        for horizon in (1, 12, 36):
            for clock in ('early', 'late'):
                rows = [v for v in r['clock_rows'] if v['mode'] == 'day_mse' and v['split'] == split and v['horizon'] == horizon and v['clock'] == clock]
                md.append(f"| {split} | {horizon} | {clock} | {100*avg(rows, 'recall'):.2f}% | {100*avg(rows, 'fpr'):.2f}% | {avg(rows, 'auc'):.4f} | {avg(rows, 'nh4_mae'):.3f} |")
    md += ['', 'early는04:05~13:00, late는13:05~23:55이다. 시간대별 상태/운전 조건/클래스 비율이 다르다. 고정 예측 길이의 시간대 연관성을 보여주며 시각 자체의 인과 효과를 입증하지 않는다.', '',
           '## 해석과 다음 범위', '',
           '- 같은 목표 시각에서도 예측 길이에 따라 성능이 크게 달라져 누적 상태/전이 오차의 관여를 지지한다. 외부 day_mse의5분 예측은 recall97.01%, FPR1.36%지만1시간에서69.73%/10.74%,3시간에서63.36%/25.34%다. 후반 시간대가 단기 예측부터 전혀 불가능한 것은 아니다.',
           '- NH4 시작 상태만 실측으로 바꾸면 day_mse 외부 recall53.78→62.71%, FPR33.28→24.33%로 개선된다. DO만 바꾸면 FPR34.18%, MAE1.666으로 악화한다. 세 모델에서 같은 방향이 관찰됐다. NH4 정보가 더 직접적인 개선 단서이지만 NH4/DO 상호작용 때문에 기여율을 독립 합산하지 않는다.',
           '- 정확한 두 시작 상태를 제공해도 day_mse의3시간 후반 FPR25.34%는 short_mse14.33%, start_multi10.92%보다 높다. 초기 편향만으로 설명되지 않는 반복 전이의 악화도 남는다. 하루 MSE를 채택하거나 관측 갱신만으로 해결됐다고 하지 않는다.',
           '- 고정3시간 예측에서도 day_mse의외부 early FPR3.55%, late25.34%이고 AUC.8435→.7676이다. 시간대와 연결된 상태/운전 조건 차이도 남는다. 누적 오차와 조건 차이 중 하나만 원인이라고 결론짓지 않는다.',
           '- 다음 보완의 표적은 NH4의 반복 전이에서 정상/위반 구분을 보존하는 것이다. 하루 MSE/평균 편향 제거/DO 보정 확대를 반복하지 않는다. 전이의 국소 오류와 생성 상태에서의 오차 증폭을 확인해 해당 기전을 고치는 후보를 정해야 한다. 새 모델이나 정책 학습은 아직 시작하지 않았다.', '',
           '기존 e27 예측 완전일치, e26 한 스텝 판정 수 일치, 무주입36전이 재생 일치, 두 상태 주입 대조 일치, 시작 이후 정답 교란 불변, 해시 불변 검사 통과. 내부/외부는 재사용한 개발 자료다. 임계값/정책/환경을 교체하지 않았고 실측 관측이 없는 가상 제어 경로에는 이 주입을 사용할 수 없다.']
    (out/'results.md').write_text('\n'.join(md)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in r['config']['modes']:
        rows = [v for v in r['rows'] if v['mode'] == mode and v['split'] == 'val']
        for ax, key in zip(axes, ('fpr', 'recall')):
            vals = [100*avg([v for v in rows if v['horizon'] == h], key) for h in r['config']['horizons']]
            ax.plot(np.array(r['config']['horizons'])*5, vals, marker='o', label=mode)
            ax.set_xscale('log')
            ax.set(xlabel='Forecast horizon (minutes)', ylabel=key+' (%)')
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Same late target times, measured origin states, validation')
    fig.tight_layout()
    (out/'figures').mkdir(exist_ok=True)
    fig.savefig(out/'figures/horizon.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.out)
