"""e27 고정 진단의 표, 판정과 그림을 생성한다."""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build(source, out):
    r = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    avg = lambda rows, k: float(np.mean([v[k] for v in rows]))
    rows = r['corrected']
    gates = []
    for mode in r['config']['modes']:
        for method in r['config']['methods'][1:]:
            checks = []
            for split in ('inner_oof', 'val'):
                for seed in (0, 1):
                    select = lambda method: next(v for v in rows if v['mode'] == mode and v['method'] == method
                                                  and v['split'] == split and v['seed'] == seed)
                    b, c = select('none'), select(method)
                    checks.append(dict(split=split, seed=seed,
                        recall_not_worse=c['recall'] >= b['recall'], fpr_lower=c['fpr'] < b['fpr'],
                        cost_mae_not_worse=c['cost_mae'] <= b['cost_mae'],
                        absolute_recall80_fpr10=c['recall'] >= .8 and c['fpr'] <= .1))
            gates.append(dict(mode=mode, method=method, checks=checks,
                         passed=all(v[k] for v in checks for k in ('recall_not_worse', 'fpr_lower', 'cost_mae_not_worse'))))
    r['candidate_screen'] = gates
    # 날짜별 보정 값과 반복 날짜 목록은 runtime 원시 JSON에 보존한다.
    summary = copy.deepcopy(r)
    for row in summary['corrected']:
        row.pop('correction_by_day')
    for row in summary['calibration']:
        for key in ('cal_days', 'eval_days'):
            dates = row.pop(key)
            row[key+'_range'] = dict(first=dates[0], last=dates[-1], count=len(dates))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    md = ['# e27 결과', '', f"실행 커밋 `{r['commit']}`, seed0,1 평균. 새 모델 학습과 test 사용 없음.",
          '보정 후처리 및 분류 진단이다. 실제 제어 성능이나 환경 교체 적격성 결과가 아니다.', '',
          '## 구분 능력', '',
          '동일 recall의 FPR은 평가 정답으로 얻은 ROC 사후 진단이다. 이 임계값을 배포 성능으로 보고하지 않는다.', '',
          '| 구간 | 모델 | AUC | FPR @ recall50 | FPR @ recall80 |',
          '| --- | --- | --- | --- | --- |']
    for split in ('inner', 'val'):
        for mode in r['config']['modes']:
            rr = [v for v in r['raw'] if v['mode'] == mode and v['split'] == split]
            fpr = lambda target: 100*np.mean([v['oracle_at_recall'][target]['fpr'] for v in rr])
            md.append(f"| {split} | {mode} | {avg(rr, 'auc'):.4f} | {fpr('0.5'):.2f}% | {fpr('0.8'):.2f}% |")
    md += ['', '## 과거 자료에서 고정한 분류 운영점의 외부 전이', '',
           '내부66일 calibration에서 운영점을 고정하고 외부43일에 적용했다. NH4/DO 물리 기준과 비용 계산은 변경하지 않았다.', '',
           '| 모델 | calibration 목표 | 실제 val recall | 실제 val FPR |',
           '| --- | --- | --- | --- |']
    for mode in r['config']['modes']:
        for point in ('recall50', 'recall80', 'fpr10'):
            rr = [v['evaluation'] for v in r['calibration'] if v['mode'] == mode and v['split'] == 'val'
                  and v['operating_point'] == point]
            md.append(f"| {mode} | {point} | {100*avg(rr, 'recall'):.2f}% | {100*avg(rr, 'fpr'):.2f}% |")
    md += ['', '## NH4 과거 잔차 보정', '',
           'inner_oof는 내부66일 중 시간순 두 fold의 평가42일을 합쳤다. 고정 평균/구간 평균/일간 EWMA는 NH4에만 적용했다. EWMA는 하루 전체 결과가 관측된 뒤 다음 날부터 갱신한다.', '',
           '| 구간 | 모델 | 보정 | recall | FPR | 하루 비용 MAE |',
           '| --- | --- | --- | --- | --- | --- |']
    for split in ('inner_oof', 'val'):
        for mode in r['config']['modes']:
            for method in r['config']['methods']:
                rr = [v for v in rows if v['mode'] == mode and v['method'] == method and v['split'] == split]
                md.append(f"| {split} | {mode} | {method} | {100*avg(rr, 'recall'):.2f}% | {100*avg(rr, 'fpr'):.2f}% | {avg(rr, 'cost_mae'):.2f} |")
    md += ['', f"후속 후보 판정 통과: {sum(v['passed'] for v in gates)}/{len(gates)}. 판정은 두 seed 각각 내부/외부에서 recall 유지, FPR 감소, 비용MAE 비악화다.",
           '', '## 하루 MSE의 예측 거리별 오류', '',
           '| 구간 | 예측 전이 | AUC | FPR | 정상 상태 편향 | 위반 상태 편향 | 전체 FP 중 비율 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for split in ('inner', 'val'):
        for k, segment in enumerate(('1~36', '37~144', '145~275')):
            rr = [v for v in r['conditions'] if v['mode'] == 'day_mse' and v['split'] == split and v['group'] == f'horizon{k}']
            fraction = np.mean([v['fp']/next(b['fp'] for b in r['raw'] if b['mode'] == 'day_mse'
                               and b['split'] == split and b['seed'] == v['seed']) for v in rr])
            md.append(f"| {split} | {segment} | {avg(rr, 'auc'):.4f} | {100*avg(rr, 'fpr'):.2f}% | {avg(rr, 'negative_bias'):+.3f} | {avg(rr, 'positive_bias'):+.3f} | {100*fraction:.2f}% |")
    md += ['', '## 결과 해석', '',
           'day_mse는 외부에서 동일 recall50의 FPR도 short_mse보다 높았다. 전체 예측의 일률적인 위치 이동만으로 설명되지 않는 구분 능력 저하다. 외부의 후반145~275전이에 오탐이 집중되고 AUC가 낮다. 정상에서는 과대, 위반에서는 과소 예측하는 경향이 함께 있어 평균 편향을 빼면 실제 위반 탐지도 줄어든다. 조건부 편향은 예측 압축과 일치하는 진단이나, 물리/최적화의 인과 기전을 확정한 것은 아니다.', '',
           '외부 recall80에서 필요한 FPR은 세 모델의 seed평균41.89~43.53%다. 평가 정답으로 전역 임계값을 골라도 기존 recall80/FPR10 목표를 달성하지 못한다. 임계값 조정만으로 충분하지 않다는 직접 근거다.', '',
           '매일01:00에서 예측을 시작하므로 예측 거리와 실제 시각이 함께 바뀐다. 후반145~275전이는13:05~23:55에 해당한다. 따라서 후반 악화를 예측 길이만의 인과 효과로 단정하지 않는다. 누적 오차와 특정 시간대의 운전 조건 영향을 분리할 필요가 있다.', '',
           '모든 단순 보정은 후보 판정에 실패했다. OMPB 학습과 환경 교체, e25 정책과의 결합은 시작하지 않았다. 다음 검토는 긴 예측 후반의 구분 능력 저하를 겨냥해야 한다. 실제 미래 상태 주입이나 기존 실패 설정의 벌점 강화로 성공을 만들지 않는다.', '',
           '날짜 paired bootstrap의 차이와95%구간, seed별 수치, 조건별 분모와 편향은 summary.json에 있다. 날짜 재표집은 시계열 의존을 완전히 처리하지 못하며 새 seed 일반화 구간이 아니다. 반복 사용한 내부/외부 개발 자료이며 최종 test가 아니다.', '',
           '검증: 기존 e14/e26 score 완전일치, 미래 정답 비참조, 당일 보정의 인과 순서, DO 불변, 동점 임계값과 보정 부호, 입력/모델/캐시/소스 해시 불변을 확인했다.']
    (out/'results.md').write_text('\n'.join(md)+'\n')
    (out/'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    markers = {'none': 'o', 'global': 'v', 'segment': 's', 'ewma_segment': 'D'}
    for ax, split in zip(axes, ('inner_oof', 'val')):
        for color, mode in zip(('C0', 'C1', 'C2'), r['config']['modes']):
            for method in r['config']['methods']:
                rr = [v for v in rows if v['mode'] == mode and v['method'] == method and v['split'] == split]
                ax.scatter(100*avg(rr, 'fpr'), 100*avg(rr, 'recall'), color=color, marker=markers[method],
                           label=f'{mode}/{method}', s=55)
        ax.axvline(10, color='gray', linestyle=':')
        ax.axhline(80, color='gray', linestyle=':')
        ax.set(xlabel='False positive rate (%)', ylabel='Recall (%)', title=split, ylim=(0, 85), xlim=(0, 21))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=6, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig(out/'figures/tradeoff.png', dpi=160)
    plt.close(fig)
    print('Candidate screen:', sum(v['passed'] for v in gates), '/', len(gates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    build(a.source, a.out)
