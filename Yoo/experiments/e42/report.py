"""미리 정한 모든 군과 주/보조 지표를 표로 만든다."""
import argparse
import json
from pathlib import Path


def main(out):
    c=json.loads((out/'comparison.json').read_text());m=json.loads((out/'manifest.json').read_text())
    lines=['# e42 정상/위반 예측기 분리 결과','',f"실행 `{m['code_commit']}`, 총 {m['elapsed_seconds']:.2f}초. NH4 학습기 25개, 새 RL/test 없음.",'',
           'control/augmented는 e41 동결 기준, direct는 다음 절대 농도 단일 회귀 T, capacity는 같은 단일 회귀 3T, split은 정상/위반/혼합기 각각 T다. 주 비교는 split 대 direct/capacity다.','']
    for mode,title in (('cold','검증 40일 95전이'),('holdout','holdout 48일'),('warm','검증 첫 1시간 실측 뒤 91전이'),('one_step','검증 한 스텝 실측 입력')):
        lines += [f'## {title}','','| 군 | NH4 MAE | NH4 recall % | NH4 FPR % | DO1 MAE | DO2 MAE | DO3 MAE | DO recall % | DO FPR % |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name,rr in c['results'].items():
            v=rr[mode];cf=v['confusion'];e=v['mae']
            lines.append(f"| {name} | {e['nh4']:.4f} | {cf['NH4']['recall_pct']:.3f} | {cf['NH4']['fpr_pct']:.3f} | {e['do1']:.4f} | {e['do2']:.4f} | {e['do3']:.4f} | {cf['DO']['recall_pct']:.3f} | {cf['DO']['fpr_pct']:.3f} |")
        lines += ['']
    lines += ['## 검증 정상/위반 구간과 초과 집계','','| 군 | 정상 MAE | 위반 MAE | 정상 편향 | 위반 편향 | NH4 초과 수 평균 | 초과 수 MAE | 초과 강도 MAE mg h/L |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,rr in c['results'].items():
        v=rr['cold'];g=c['group_errors'][name]['cold']
        lines.append(f"| {name} | {g['normal']['mae']:.4f} | {g['violation']['mae']:.4f} | {g['normal']['bias']:+.4f} | {g['violation']['bias']:+.4f} | {v['counts']['NH4']['predicted_mean']:.3f} | {v['counts']['NH4']['mae']:.3f} | {v['nh4_excess_mg_h_L']['mae']:.3f} |")
    lines += ['','실측 NH4 초과 수 평균은 16.45다. 실제 정상/위반 구분은 평가 집계에만 쓰며 추론 전문가 선택에는 쓰지 않는다. MAE와 편향 단위는 mg/L다.','',
              '## 혼합기의 진단','','| 구간 | AUC | Brier | 실제 정상 평균 g | 실제 위반 평균 g | 실제 위반에서 mu0 | 실제 위반에서 mu1 |','| --- | --- | --- | --- | --- | --- | --- |']
    for mode,v in c['gate_diagnostics'].items():
        lines.append(f"| {mode} | {v['auc']:.4f} | {v['brier']:.4f} | {v['mean_probability_normal']:.4f} | {v['mean_probability_violation']:.4f} | {v['mean_normal_expert_on_violations']:.4f} | {v['mean_violation_expert_on_violations']:.4f} |")
    lines += ['','혼합기 확률은 별도 calibration으로 보정하지 않았다. g>0.5 진단은 JSON에만 있으며 주 비용/판정을 대체하지 않는다.','',
              '## 공기 변화 뒤 반응','','| 군 | 구간 | 분 | 사건 수 | NH4 MAE | 실제 변화 | 예측 변화 | recall % | FPR % |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,rows in c['events'].items():
        for v in rows:
            cf=v['confusion'];r='정의 불가' if cf['recall_pct'] is None else f"{cf['recall_pct']:.3f}";f='정의 불가' if cf['fpr_pct'] is None else f"{cf['fpr_pct']:.3f}"
            lines.append(f"| {name} | {v['group']} | {v['minutes']} | {v['windows']} | {v['nh4_mae']:.4f} | {v['nh4_actual_delta']:+.4f} | {v['nh4_predicted_delta']:+.4f} | {r} | {f} |")
    lines += ['','decrease_rising은 실측 NH4가 0.1보다 상승한 공기 감소 구간이다. 관측 재생의 부분집합이며 인과 효과가 아니다.','',
              '## split에서 비교군을 뺀 변화','','| 비교군 | 지표 | 차이 | 날짜 7일 블록 95% 구간 |','| --- | --- | --- | --- |']
    for name,v in c['uncertainty_split_minus'].items():
        for target,row in v['mae'].items():
            a,b=row['paired_7day_95ci'];lines.append(f"| {name} | {target} MAE | {row['difference']:+.4f} | [{a:+.4f}, {b:+.4f}] |")
        for target,rows in v['confusion'].items():
            for key,row in rows.items():
                a,b=row['paired_7day_95ci_pp'];lines.append(f"| {name} | {target} {key} pp | {row['difference_pp']:+.3f} | [{a:+.3f}, {b:+.3f}] |")
    lines += ['','2000회, seed4100. 반복 개발 검증의 탐색적 구간이며 독립 확증이나 비열등성 검정이 아니다.','',
              '## 학습량과 표본','','| 군 | 구성원 | 트리 수 | 총 행 수 | 행-트리 수 | 초 |','| --- | --- | --- | --- | --- | --- |']
    for v in c['training']:
        lines.append(f"| {v['arm']} | {v['member']} | {v['trees']} | {v['rows']} | {v['row_tree_work']} | {v['seconds']:.3f} |")
    lines += ['','capacity와 split은 합계 트리 수가 같지만 학습할 행 수와 벽시계 시간은 다르다. 클래스별 학습 행/날짜 수는 comparison.json에 기록한다.','',
              '## 사전 선별','','| 비교군 | 항목 | 통과 |','| --- | --- | --- |']
    for name,checks in c['screens_split_vs'].items():
        for key,value in checks.items():lines.append(f"| {name} | {key} | {'예' if value else '아니오'} |")
    for key,value in c['preservation_screen'].items():lines.append(f"| 기존 장점 보존 | {key} | {'예' if value else '아니오'} |")
    lines += ['',f"전체 통과: {c['all_screen_pass']}. 자동 채택 없음. 상세 해석은 README.md에 기록한다.",'']
    (out/'results.md').write_text('\n'.join(lines));print(out/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
