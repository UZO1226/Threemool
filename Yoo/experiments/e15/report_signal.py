"""초기 오차의 조건부 예측력 검사 결과를 표와 그림으로 보존한다."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from initial_signal import sha, write, signal_gate


def main(a):
    r = json.loads(a.signal.read_text())
    assert 'completed_at' in r and all(r['checks'].values())
    assert r['gate'] == signal_gate(r['seeds'])
    for path, expected in r['model_sha256'].items(): assert sha(path) == expected
    assert not r['inner_holdout_used'] and not r['outer_val_used'] and not r['test_used']
    for seed in r['seeds']:
        for mode, rows in seed['oof_predictions'].items():
            assert len(rows) == 129
            np.testing.assert_allclose(np.mean([(x['prediction']-x['target'])**2 for x in rows]), seed['pooled_mse'][mode], rtol=1e-12)
    summary = copy.deepcopy(r)
    for row in summary['seeds']:
        # 초기 실측 특징 배열은 서버에만 두고 회귀 계수와 파생 오차를 보존한다.
        del row['initial_features']
        del row['target_daily_bias']
        del row['target_bias_by_segment']
    summary['full_report_sha256'] = sha(a.signal)
    a.out.mkdir(exist_ok=True, parents=True)
    write(a.out/'signal_summary.json', summary)
    md = ['# Yoo/e15 초기 오차 예측력 검사 결과', '', f"실행 `{r['commit']}`.",
          f"시작 `{r['started_at']}`, 완료 `{r['completed_at']}` (UTC).", '',
          '기반 모델은 e14 day_mse seed 0, 1로 고정했다. fit 263일 안의 회귀 시간 분할 3개, 날짜 단위 확인 예측 129개를 집계했다. 기반 모델은 fit 전체를 이미 학습했으므로 전체 시스템의 독립 검증으로 부르지 않는다. 내부 holdout 66일, 외부 val, test는 사용하지 않았다.', '',
          '## 이후 NH4 평균 편향을 예측하는 MSE', '',
          '목표는 01:05부터 23:55까지 모델 예측 농도와 실측 농도의 차이를 평균한 값이다. 아래 값은 그 평균 편향을 예측한 회귀 오차다. 수질 예측 MAE나 거짓양성률이 아니다.', '',
          '| 회귀 입력 | seed 0 MSE | seed 1 MSE |', '| --- | --- | --- |']
    labels = dict(constant='평균 편향만', context='초기 조건 6개', residual='초기 평균 잔차 2개', context_residual='초기 조건 및 잔차')
    for mode, label in labels.items():
        values = [s['pooled_mse'][mode] for s in r['seeds']]
        md.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    md += ['', '## 초기 조건 대비 잔차의 추가 효과', '',
           '| seed | 비교 기준 | MSE 감소율 | 더 나은 분할 수 | 진행 기준 |', '| --- | --- | --- | --- | --- |']
    for c in r['gate']['checks']:
        md.append(f"| {c['seed']} | {labels[c['base']]} | {c['pooled_gain']*100:+.2f}% | {c['better_folds']}/3 | {c['passed']} |")
    md += ['', '감소율이 음수이면 오차가 증가했다는 뜻이다. 기준은 두 seed 각각에서 평균 편향 및 초기 조건 양쪽 대비 MSE 5% 이상 감소, 세 분할 중 두 개 이상 개선이다.', '',
           '## 시간 분할별 MSE', '', '| seed | 분할 | 회귀 학습 날짜 수 | 확인 날짜 수 | 평균만 | 초기 조건 | 초기 잔차 | 조건 및 잔차 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in r['seeds']:
        for f in s['folds']:
            md.append(f"| {s['seed']} | {f['fold']+1} | {f['train_days']} | {f['test_days']} | " + ' | '.join(f"{f['mse'][m]:.4f}" for m in labels)+' |')
    md += ['', '## 판정', '', f"사전 진행 기준 통과: **{r['gate']['passed']}**.", '',
           '초기 평균 잔차를 더한 선형 회귀는 두 seed의 세 분할 모두 초기 조건만 사용한 회귀보다 나빴다. 이 입력 구성의 편향 보정기를 추가할 근거를 확보하지 못했다. 사전 계획대로 보정기 학습 및 내부 holdout 평가를 진행하지 않았다. 기반 모델과 FCSRL + PPO는 유지했다.', '',
           '이 결과는 초기 12전이 평균 잔차 2개와 고정 ridge 회귀에 대한 것이다. 모든 시간 이력, 비선형 추정기, 실제 센서 피드백이 무효라는 뜻은 아니다. 초기 조건만 쓰는 회귀의 이득을 근거로 사후에 주 후보를 바꾸지 않았다.', '',
           '![초기 오차의 추가 예측력](figures/signal_mse.png)', '']
    (a.out/'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    for i, s in enumerate(r['seeds']):
        ax.bar(x + (i-.5)*.35, [s['pooled_mse'][m] for m in labels], width=.35, label=f"Seed {s['seed']}")
    ax.set_xticks(x, ['Constant', 'Initial context', 'Initial residual', 'Context + residual'])
    ax.set(ylabel='MSE of predicted daily NH4 bias', title='Forward regression validation within base-model fit days')
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    (a.out/'figures').mkdir(exist_ok=True)
    fig.savefig(a.out/'figures/signal_mse.png', dpi=160)
    plt.close(fig)
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('signal', 'out'): p.add_argument('--'+key, type=Path, required=True)
    main(p.parse_args())
