"""e66 test 최종 평가 집계: J1 ~ J5."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VAL = {s: Path('/home/jio/fcsrl_src/results/e64/main/runs') if s < 5 else Path('/home/jio/fcsrl_src/results/e65/main/eval') for s in range(10)}
LABEL = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL'}
TAGS = {'corrected_cap1_daily': 'T1 DO 수정 모델, 일별, 상한 (주 평가)', 'common_cap1_daily': 'T2 공통 v1.1 모델, 일별, 상한',
        'corrected_cap1_continuous': 'T3 DO 수정 모델, test 41일 연속, 상한'}


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def verdict(rows):
    both = sum(r['air_m3'] < 0 and r['cost'] < 0 for r in rows)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    if both >= 8 and mean['air_m3'] < 0 and mean['cost'] < 0:
        v = 'test 에서 배분 이득 유지'
    elif mean['cost'] < 0 and mean['air_m3'] > 0:
        v = '교환 (공기 증가)'
    elif mean['air_m3'] < 0 and mean['cost'] > 0:
        v = '교환 (비용 증가)'
    else:
        v = '이득 없음'
    return both, mean, v


def main(a):
    R = {}
    for p in a.root.glob('runs/*/*/result.json'):
        R[(p.parent.parent.name, p.parent.name)] = json.loads(p.read_text())['test']
    L = ['# e66 결과: test 41일 최종 평가', '', 'test 41일 (2021-12-20 ~ 2022-01-29) x 구성원 5개, 한 번만 실행. 정책은 e63 (seed 0 ~ 4) 과 e65 (seed 5 ~ 9). 모든 정책은 park_e19 차단과 실행 상한. 대리 환경 결과이며 현장 성능이 아니다.', '']
    J = {}
    head = ['| 방법 | 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | NH4 초과 mg h/L | 최악 날짜 NH4 | 상한으로 잘린 스텝 % |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for tag, title in TAGS.items():
        L += [f'## {title}', ''] + head
        for m in ('ppo', 'fcsrl'):
            for c in ('total', 'alloc'):
                v = [R[(tag, f'{c}_{m}_s{s}')] for s in range(10)]
                L.append(f"| {LABEL[m]} | {'총 공기만' if c == 'total' else '총 공기 + 배분'} | {ms([x['air_m3'] for x in v], 0)} | {ms([x['nh4_cost'] for x in v])} | "
                         f"{ms([x['do_cost'] for x in v])} | {ms([x['cost'] for x in v])} | {ms([x['nh4_excess_mg_h_L'] for x in v])} | "
                         f"{ms([x['worst_day_nh4_cost'] for x in v], 1)} | {ms([x['clipped_pct'] for x in v], 1)} |")
        for ref, lab in (('record', '기록 운전 (차단 없음)'), ('zero', '공기 0 + park_e19')):
            x = R[(tag, ref)]
            L.append(f"| {lab} | | {x['air_m3']:,.0f} | {x['nh4_cost']:.2f} | {x['do_cost']:.2f} | {x['cost']:.2f} | {x['nh4_excess_mg_h_L']:.2f} | {x['worst_day_nh4_cost']:.1f} | {x['clipped_pct']:.1f} |")
        L += ['', '| 방법 | 둘 다 낮은 seed | 평균 공기 변화 | 평균 비용 변화 | 평균 NH4 변화 | 평균 DO 변화 | 판정 |', '| --- | --- | --- | --- | --- | --- | --- |']
        for m in ('ppo', 'fcsrl'):
            rows = [{k: R[(tag, f'alloc_{m}_s{s}')][k] - R[(tag, f'total_{m}_s{s}')][k] for k in ('air_m3', 'cost', 'nh4_cost', 'do_cost')} for s in range(10)]
            both, mean, v = verdict(rows)
            J[f'{tag}|{m}'] = dict(paired=rows, mean=mean, both_lower_seeds=both, verdict=v)
            L.append(f"| {LABEL[m]} | {both}/10 | {mean['air_m3']:+,.0f} | {mean['cost']:+.3f} | {mean['nh4_cost']:+.3f} | {mean['do_cost']:+.3f} | {v} |")
        L.append('')
    L += ['## J5 검증 대비 test (DO 수정 모델, 일별, 상한, seed 0 ~ 9 평균)', '', '| 방법 | 조건 | 검증 공기 / 비용 | test 공기 / 비용 |', '| --- | --- | --- | --- |']
    for m in ('ppo', 'fcsrl'):
        for c in ('total', 'alloc'):
            val = [json.loads((VAL[s] / 'corrected_cap1_daily' / f'{c}_{m}_s{s}' / 'result.json').read_text())['val'] for s in range(10)]
            te = [R[('corrected_cap1_daily', f'{c}_{m}_s{s}')] for s in range(10)]
            L.append(f"| {LABEL[m]} | {'총 공기만' if c == 'total' else '총 공기 + 배분'} | {np.mean([x['air_m3'] for x in val]):,.0f} / {np.mean([x['cost'] for x in val]):.2f} | "
                     f"{np.mean([x['air_m3'] for x in te]):,.0f} / {np.mean([x['cost'] for x in te]):.2f} |")
    L += ['', '## 종합 (사전 규칙)', '']
    for m in ('ppo', 'fcsrl'):
        L.append(f"- {LABEL[m]}: T1 {J[f'corrected_cap1_daily|{m}']['verdict']}, T2 {J[f'common_cap1_daily|{m}']['verdict']}, T3 {J[f'corrected_cap1_continuous|{m}']['verdict']}.")
    L += ['', 'test 는 이 실행에서 한 번만 썼다. 결과를 본 뒤 설정이나 정책을 바꿔 다시 평가하지 않는다.']
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(dict(judgement=J, root=str(a.root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
