"""e62 집계: A 분해 표, B 배분 표, 사전 규칙 판정."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = {'common_v1_1': '공통 v1.1', 'do_corrected': 'DO 수정 (e51)'}


def main(a):
    r = json.loads((a.root / 'result.json').read_text())
    A, S, B, D = r['A'], r['B_support'], r['B_rollouts'], r['B_data']
    L = ['# e62 결과: NH4 바닥 분해와 구간 배분 진단', '',
         'Kaggle DARROW, e50 준비물. 새 정책 학습 없음, test 없음. 검증 40일 x 구성원 5개 (A, B), 학습 188일 x 구성원 5개 (B).', '',
         '## A. 차단 아래 NH4 비용 분해 (검증, 공통 v1.1 모델, seed 평균)', '',
         '시작 이월은 첫 스텝부터 끊기지 않고 이어진 NH4>4 스텝 수다. 바닥은 같은 에피소드에서 총 공기 9,000 고정, 기록 배분의 NH4 비용이다.', '',
         '| 조건 | NH4 비용 | 시작 이월 | 제어 구간 | 바닥 초과 | 시작 NH4>4 에피소드의 비용 | 시작 NH4<=4 에피소드의 비용 |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for name, v in A.items():
        label = '총 공기 9,000 고정 (바닥)' if name == 'floor_max_air' else name
        L.append(f"| {label} | {v['nh4_cost']:.2f} | {v['carryover']:.2f} | {v['controllable']:.2f} | {v['above_floor']:+.2f} | "
                 f"{v['nh4_cost_start_above4']:.2f} | {v['nh4_cost_start_below4']:.2f} |")
    L.append(f"\n시작 NH4>4 인 검증 에피소드 비율 {A['floor_max_air']['share_episodes_start_above4'] * 100:.1f}%.")
    L += ['', '## B1. 기록 자료의 배분 범위 (train)', '',
          '| 구간 | 공기 99% 분위수 m³/h | 최댓값 | 비율 5% | 비율 50% | 비율 95% |', '| --- | --- | --- | --- | --- | --- |']
    for i in range(5):
        L.append(f"| air{i + 1} | {S['zone_air_p99'][i]:,.0f} | {S['zone_air_max'][i]:,.0f} | {S['fraction_quantiles']['5'][i]:.3f} | "
                 f"{S['fraction_quantiles']['50'][i]:.3f} | {S['fraction_quantiles']['95'][i]:.3f} |")
    L.append(f"\n뒤 구간 비율 (air4+air5) 분위수: " + ', '.join(f"{q}% {v:.3f}" for q, v in S['back_share_quantiles'].items()))
    for split in ('val', 'train'):
        L += ['', f'## B2. 배분별 결과 ({split})', '',
              '| 모델 | 일정 | 배분 | NH4 비용 | DO 비용 | 합산 비용 | 공기 m³ | coverage p99 | coverage max |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for key, v in B.items():
            m, sp, sch, al = key.split('|')
            if sp != split:
                continue
            L.append(f"| {MODEL[m]} | {'9,000 고정' if sch == 'max9000' else '기록 총 공기'} | {al} | {v['nh4_cost']:.2f} | {v['do_cost']:.2f} | {v['cost']:.2f} | "
                     f"{v['air_m3']:,.0f} | {v['coverage_p99']:.3f} | {v['coverage_max']:.3f} |")
    L += ['', '## B3. 기록 자료와 모델의 뒤 구간 비율 계수 (다음 스텝 NH4 변화량, 비율 1 단위당 mg/L)', '',
          f"train {D['days']}일 {D['rows']:,}행, 날짜 묶음 bootstrap. 뒤 구간 비율 평균 {D['share_mean']:.3f}, 표준편차 {D['share_sd']:.3f}.", '',
          '| 대상 | 계수 | 95% 구간 |', '| --- | --- | --- |',
          f"| 기록 자료 | {D['data_coef']:+.4f} | [{D['data_ci95'][0]:+.4f}, {D['data_ci95'][1]:+.4f}] |",
          f"| 공통 모델 예측 | {D['model_coef']:+.4f} | [{D['model_ci95'][0]:+.4f}, {D['model_ci95'][1]:+.4f}] |"]
    # 판정
    allocs = sorted({k.split('|')[3] for k in B} - {'record'})
    c1 = {}
    for al in allocs:
        ok = True
        for m in MODEL:
            v, rec = B[f'{m}|val|max9000|{al}'], B[f'{m}|val|max9000|record']
            ok &= v['coverage_p99'] >= 0.95 and rec['nh4_cost'] - v['nh4_cost'] >= 0.3
        c1[al] = bool(ok)
    c2 = {al: B[f'do_corrected|val|record|{al}']['cost'] < B['do_corrected|val|record|record']['cost'] for al in allocs}
    passing = [al for al in allocs if c1[al] and c2[al]]
    c3 = D['data_coef'] < 0 and D['data_ci95'][1] < 0 and D['model_coef'] < 0
    go = bool(passing and c3)
    L += ['', '## 사전 규칙 판정', '',
          '| 배분 | (1) 두 모델 바닥 0.3 이상 감소, coverage 0.95 이상 | (2) DO 수정 모델 기록 총 공기 재생 합산 비용 감소 |', '| --- | --- | --- |']
    L += [f'| {al} | {c1[al]} | {c2[al]} |' for al in allocs]
    L += ['', f'(3) 기록 자료 계수 음수, 구간 0 미포함, 모델 계수 음수: {c3}', '',
          f"(1)(2) 를 함께 만족하는 배분: {', '.join(passing) if passing else '없음'}. 종합: {'배분 정책 실험으로 진행' if go else '배분 방향 중단'}.",
          '', '반복 사용한 개발 val 의 진단이며 현장 효과를 주장하지 않는다.']
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(dict(criterion1=c1, criterion2=c2, criterion3=bool(c3), passing=passing, go=go,
                                                       A=A, B_data=D, root=str(a.root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
