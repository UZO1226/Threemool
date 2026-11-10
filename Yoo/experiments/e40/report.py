"""저장된 수치에서 같은 프로필의 보정/연속 점수 표를 만든다."""
import argparse
import json
from pathlib import Path


def main(out):
    result = json.loads((out / 'comparison.json').read_text())
    operating = json.loads((out / 'operating_points.json').read_text())['models']
    manifest = json.loads((out / 'manifest.json').read_text())
    lines = ['# e40 연속 NH4 점수 판정 결과', '',
             f"실행 커밋 `{manifest['code_commit']}`, 평가 {manifest['elapsed_seconds']:.3f}초. 새 모델/정책 학습 없음.", '',
             '보정 48일에서 경계를 고정하고 검증 40일 x 5구성원 x 95전이에 적용했다. 실제 NH4>4 정의를 유지한다. 검증은 반복 사용한 개발 자료다.', '',
             '## 보정 자료에서 선택한 경계', '',
             '| 모델 | 목표 프로필 | 점수 경계 | 보정 탐지율 % | 보정 FPR % |',
             '| --- | --- | --- | --- | --- |']
    for model, methods in operating.items():
        for mode, v in methods.items():
            boundary = '경보 없음' if v['threshold'] is None else f"{v['threshold']:.6f}"
            lines.append(f"| {model} | {mode} | {boundary} | {v['recall_pct']:.3f} | {v['fpr_pct']:.3f} |")
    lines += ['', '위 점수 경계는 경보를 고르는 값이며 실제 수질 기준이나 재귀 상태를 바꾸지 않는다.', '',
              '## 검증 결과', '', '| 모델 | 판정 | 탐지율 % | FPR % | 예측 초과 표본/일/구성원 |',
              '| --- | --- | --- | --- | --- |']
    for model, methods in result['validation'].items():
        for method, v in methods.items():
            lines.append(f"| {model} | {method} | {v['recall_pct']:.3f} | {v['fpr_pct']:.3f} | {v['predicted_count_per_day_member']:.3f} |")
    lines += ['', '실측 평균 초과 표본 수는 16.45다. 예측 평균 수가 비슷해도 위반 시점을 잘 탐지했다는 뜻은 아니다.', '',
              '## 연속 판정에서 isotonic 판정을 뺀 차이', '',
              '| 모델 | 프로필 | 탐지율 차이 pp [95% CI] | FPR 차이 pp [95% CI] |',
              '| --- | --- | --- | --- |']
    def fmt(v):
        a, b = v['paired_7day_block_95ci_pp']
        return f"{v['difference_pp']:+.3f} [{a:+.3f}, {b:+.3f}]"
    for model, methods in result['continuous_minus_isotonic'].items():
        for mode, v in methods.items():
            lines.append(f"| {model} | {mode} | {fmt(v['recall_pct'])} | {fmt(v['fpr_pct'])} |")
    lines += ['', '같은 날짜의 7일 블록 bootstrap 2000회다. 사후 후속 설계와 반복 검증을 고려한 탐색적 구간이며, 독립 확증이나 비열등성 판정은 아니다.', '',
              '## 연속 판정을 쓰는 이력 모델에서 기존 모델을 뺀 차이', '',
              '| 프로필 | 탐지율 차이 pp [95% CI] | FPR 차이 pp [95% CI] |', '| --- | --- | --- |']
    for mode, v in result['history_minus_base_continuous'].items():
        lines.append(f"| {mode} | {fmt(v['recall_pct'])} | {fmt(v['fpr_pct'])} |")
    lines += ['', '두 모델의 검증 탐지율이 달라 동일 탐지율 비교가 아니다.', '',
              '## 공기 감소 후 실제 NH4 상승 구간', '',
              '| 모델 | 프로필 | 분 | 구간 수 | 실제 위반 구간 수 | 탐지율 % | FPR % |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for e in result['events']:
        if e['group'] != 'air_decrease_and_nh4_rising':
            continue
        recall = '정의 불가' if e['recall_pct'] is None else f"{e['recall_pct']:.3f}"
        fpr = '정의 불가' if e['fpr_pct'] is None else f"{e['fpr_pct']:.3f}"
        lines.append(f"| {e['model']} | {e['mode']} | {e['minutes']} | {e['windows']} | {e['positives']//5} | {recall} | {fpr} |")
    lines += ['', '미래 NH4 상승 여부는 평가 부분집합에만 쓴다. 모든 공기 변화 구간의 결과는 comparison.json에 있다. 모델 5구성원이 같은 실제 사건을 공유하므로 실제 위반 구간 수를 구성원 수로 부풀리지 않는다.', '',
              '## 판정과 경계', '',
              '이력 모델의 확률 동점으로 인한 무경보는 연속 점수에서 사라졌다. 하지만 보정 FPR 9.984%가 검증 13.514%로 늘었고 탐지율은 61.077%에서 36.079%로 낮아졌다. 주 recall80 조건도 검증 72.553%로 목표 미달이다. 두 지표 동시 우위와 기간 간 목표 유지는 달성하지 못했다.', '',
              'e39 사후 진단의 FPR 9.998%와 이번 9.984% 차이는 같은 TP에서 최소 FP 경계를 고르는 명시적 동점 처리 때문이다. 보정 TP는 둘 다 907이며 이번 FP는 2128이다. 추가 오경보 3건 없이 같은 위반을 잡는 경계를 선택했다. 검증으로 경계를 조정한 결과가 아니다.', '',
              '연속 점수 선택 함수는 정보 보존 도구로 보관하되 이번 경계를 제어 환경에 채택하지 않는다. 물리 농도 예측, DO, 공기량, 보상과 비용 한도는 e39 그대로다. 추가 보정법/경계 탐색을 반복하지 않고 NH4 전이의 공기 감소 후 상승 예측 문제로 돌아간다. 후속 모델 학습은 실행하지 않았다.', '']
    (out / 'results.md').write_text('\n'.join(lines))
    print(out / 'results.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
