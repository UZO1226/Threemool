"""e64 집계: J1 (상한 일별), J2 (연속), J3 보고, e66 주 구성 규칙."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E63 = Path('/home/jio/fcsrl_src/results/e63/main')
LABEL = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL'}
TAGS = {'corrected_cap1_daily': 'DO 수정 모델, 일별, 상한', 'common_cap1_daily': '공통 v1.1 모델, 일별, 상한',
        'corrected_cap0_continuous': 'DO 수정 모델, 연속, 상한 없음', 'corrected_cap1_continuous': 'DO 수정 모델, 연속, 상한'}


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def verdict(rows):
    both = sum(r['air_m3'] < 0 and r['cost'] < 0 for r in rows)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    if both >= 4 and mean['air_m3'] < 0 and mean['cost'] < 0:
        v = '이득 유지'
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
        R[(p.parent.parent.name, p.parent.name)] = json.loads(p.read_text())['val']
    e63 = {('corrected', c, m, s): json.loads((E63 / 'runs' / c / m / f'seed_{s}' / 'metrics.json').read_text())['val_park_e19']
           for c in ('total', 'alloc') for m in ('ppo', 'fcsrl') for s in range(5)}
    e63.update({('common', c, m, s): json.loads((E63 / 'cross' / c / m / f'seed_{s}' / 'result.json').read_text())['val']
                for c in ('total', 'alloc') for m in ('ppo', 'fcsrl') for s in range(5)})
    L = ['# e64 결과: 실행 상한과 연속 날짜 재평가', '',
         'e63 정책 20개를 새 학습 없이 다시 평가했다. 상한은 구간 공기를 train 99% 분위수 이하로 자르고 잘린 양은 총 공기에서 뺀다. 연속 평가는 구성원별로 검증 40일을 잇는다. 모든 정책은 park_e19 차단. test 없음.', '']
    head = ['| 방법 | 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | 상한으로 잘린 스텝 % | w 평균 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    J = {}
    for tag, title in TAGS.items():
        L += [f'## {title}', ''] + head
        for m in ('ppo', 'fcsrl'):
            for c in ('total', 'alloc'):
                v = [R[(tag, f'{c}_{m}_s{s}')] for s in range(5)]
                L.append(f"| {LABEL[m]} | {'총 공기만' if c == 'total' else '총 공기 + 배분'} | {ms([x['air_m3'] for x in v], 0)} | {ms([x['nh4_cost'] for x in v])} | "
                         f"{ms([x['do_cost'] for x in v])} | {ms([x['cost'] for x in v])} | {ms([x['clipped_pct'] for x in v], 1)} | {ms([x['w_mean'] for x in v], 3)} |")
        for ref, lab in (('record', '기록 운전 (차단 없음)'), ('zero', '공기 0 + park_e19')):
            x = R[(tag, ref)]
            L.append(f"| {lab} | | {x['air_m3']:,.0f} | {x['nh4_cost']:.2f} | {x['do_cost']:.2f} | {x['cost']:.2f} | {x['clipped_pct']:.1f} | |")
        L += ['', '| 방법 | alloc minus total | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | 평균 | 판정 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for m in ('ppo', 'fcsrl'):
            rows = [{k: R[(tag, f'alloc_{m}_s{s}')][k] - R[(tag, f'total_{m}_s{s}')][k] for k in ('air_m3', 'cost', 'nh4_cost', 'do_cost')} for s in range(5)]
            both, mean, v = verdict(rows)
            J[f'{tag}|{m}'] = dict(paired=rows, mean=mean, both_lower_seeds=both, verdict=v)
            L.append(f"| {LABEL[m]} | 공기 | " + ' | '.join(f"{r['air_m3']:+,.0f}" for r in rows) + f" | {mean['air_m3']:+,.0f} | {both}/5 둘 다 낮음 |")
            L.append(f"| {LABEL[m]} | 합산 비용 | " + ' | '.join(f"{r['cost']:+.3f}" for r in rows) + f" | {mean['cost']:+.3f} | {v} |")
        L.append('')
    L += ['## J3 상한 전후 비교 (일별, alloc 조건, seed 평균)', '', '| 모델 | 방법 | 상한 없음 공기 / 비용 | 상한 공기 / 비용 |', '| --- | --- | --- | --- |']
    for model in ('corrected', 'common'):
        for m in ('ppo', 'fcsrl'):
            before = [e63[(model, 'alloc', m, s)] for s in range(5)]
            after = [R[(f'{model}_cap1_daily', f'alloc_{m}_s{s}')] for s in range(5)]
            L.append(f"| {model} | {LABEL[m]} | {np.mean([x['air_m3'] for x in before]):,.0f} / {np.mean([x['cost'] for x in before]):.2f} | "
                     f"{np.mean([x['air_m3'] for x in after]):,.0f} / {np.mean([x['cost'] for x in after]):.2f} |")
    final = {m: ('alloc' if J[f'corrected_cap1_daily|{m}']['verdict'] == '이득 유지' else 'total') for m in ('ppo', 'fcsrl')}
    L += ['', '## 종합과 e66 주 구성 (사전 규칙)', '']
    for m in ('ppo', 'fcsrl'):
        L.append(f"- {LABEL[m]}: 상한 일별 DO 수정 {J[f'corrected_cap1_daily|{m}']['verdict']}, 공통 {J[f'common_cap1_daily|{m}']['verdict']}, "
                 f"연속 상한 없음 {J[f'corrected_cap0_continuous|{m}']['verdict']}, 연속 상한 {J[f'corrected_cap1_continuous|{m}']['verdict']}. "
                 f"e66 주 구성: {'총 공기 + 배분' if final[m] == 'alloc' else '총 공기만'} + park_e19 + 실행 상한.")
    L += ['', '반복 사용한 개발 val 의 재평가이며 test 는 평가하지 않았다.']
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(dict(judgement=J, final_config=final, root=str(a.root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
