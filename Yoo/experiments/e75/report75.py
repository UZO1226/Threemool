"""Descriptive report of fixed planner budgets; no optimality or independent-test claim."""
import argparse
import json
from pathlib import Path
import plan75 as P


def main(root):
    rows=[json.loads(p.read_text()) for p in sorted((root/'jobs').glob('*/result.json'))]
    assert len(rows)==16, f'incomplete: {len(rows)}/16'
    lines=['# e75 고정 대리 환경의 계획 여지 진단','',
           '12 dev 날짜와12 val 날짜, 구성원5개. 미래 외생 입력과 환경 구성원을 아는 사후 행동열 탐색이다. 전역 최적 상한, 온라인 MPC 성능 또는 현장 성능이 아니다. 모델/정책 학습 없음.','',
           '| 분할 | 목적 | 탐색 | 난수 | ABAC 공기 | 후보 공기 | 절감 % | ABAC 비용 | 후보 비용 | 비용 감소 % | NH4 초과 | DO 스텝 | 한도 충족 % |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        b,f=r['baseline'],r['final']
        air=100*(1-f['air_m3']/b['air_m3']);cost=100*(1-f['step_cost']/b['step_cost']) if b['step_cost'] else 0.
        lines.append(f"| {r['split']} | {r['mode']} | {r['algorithm']} | {r['search_seed']} | {b['air_m3']:.1f} | {f['air_m3']:.1f} | {air:.2f} | {b['step_cost']:.3f} | {f['step_cost']:.3f} | {cost:.2f} | {f['nh4_excess_mg_h_L']:.3f} | {f['do_steps']:.3f} | {f['within_4_75_pct']:.1f} |")
    lines += ['','## 탐색 예산과 구성원 이전','',
              '| 분할/목적/탐색/난수 | 2세대 공기/비용 | 8세대 공기/비용 | 이전 ABAC 공기/비용 | 이전 후보 공기/비용 | 이전 비악화 % | 소요 초 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        b,f=r['transfer_baseline'],r['transfer'];q=r['snapshots']['2'];z=r['final']
        lines.append(f"| {r['split']}/{r['mode']}/{r['algorithm']}/{r['search_seed']} | {q['air_m3']:.1f}/{q['step_cost']:.3f} | {z['air_m3']:.1f}/{z['step_cost']:.3f} | {b['air_m3']:.1f}/{b['step_cost']:.3f} | {f['air_m3']:.1f}/{f['step_cost']:.3f} | {r['transfer_weak_dominance_pct']:.1f} | {r['elapsed_seconds']:.1f} |")
    lines += ['','## 해석 제한','',
              '- 비악화는 ABAC fallback과候補 제약으로 보장한 설계 성질이다. 중요한 것은 그 조건에서 추가 개선을 찾았는지다.'.replace('候補','후보'),
              '- 작은 이득은 최적 개선 여지가 작다는 증명이 아니다. 블록 잔차와 보수적인 날짜별 제약, 제한된 탐색 예산도 원인일 수 있다.',
              '- 찾은 해는 날짜/환경 구성원을 알고 고른다. 독립 검증, RL이 학습할 수 있는 성능 또는 FCSRL 기여로 해석하지 않는다.',
              '- 다음 구성원 이전은 열린 고리 민감도와 모델 차이를 함께 반영한다. 실제 외부 환경 검증은 아니다.',
              '- 모델 구성원과 탐색 seed를 독립 정책 seed로 세지 않는다. 개발용 기술 통계이며 새 유의성 검정은 없다.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    P.write_json(root/'summary.json',dict(jobs=rows,optimality_certified=False,holdout_evaluated=False,test_evaluated=False))
    print(root/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
