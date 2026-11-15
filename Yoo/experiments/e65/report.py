"""e65 집계: seed 5 ~ 9 재현 (J1, J3) 과 seed 0 ~ 9 합산 (J2)."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E64 = Path('/home/jio/fcsrl_src/results/e64/main/runs')
LABEL = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL'}


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def paired(R, model, m, seeds):
    return [{k: R[(model, 'alloc', m, s)][k] - R[(model, 'total', m, s)][k] for k in ('air_m3', 'cost', 'nh4_cost', 'do_cost')} for s in seeds]


def judge(rows, need):
    both = sum(r['air_m3'] < 0 and r['cost'] < 0 for r in rows)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return both, mean, bool(both >= need and mean['air_m3'] < 0 and mean['cost'] < 0)


def main(a):
    R = {}
    for model in ('corrected', 'common'):
        for c in ('total', 'alloc'):
            for m in ('ppo', 'fcsrl'):
                for s in range(5, 10):
                    R[(model, c, m, s)] = json.loads((a.root / 'eval' / f'{model}_cap1_daily' / f'{c}_{m}_s{s}' / 'result.json').read_text())['val']
                for s in range(5):
                    R[(model, c, m, s)] = json.loads((E64 / f'{model}_cap1_daily' / f'{c}_{m}_s{s}' / 'result.json').read_text())['val']
    L = ['# e65 결과: seed 5 ~ 9 재현', '', 'e63 과 같은 학습, seed 만 5 ~ 9. 평가는 e64 와 같은 일별 실행 상한, park_e19, 검증 40일 x 구성원 5개. test 없음.', '']
    J = {}
    for model, title in (('corrected', 'DO 수정 모델'), ('common', '공통 v1.1 모델')):
        L += [f'## {title}, 일별, 상한', '', '| 방법 | 조건 | seed | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | w 평균 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for m in ('ppo', 'fcsrl'):
            for c in ('total', 'alloc'):
                for seeds, lab in ((range(5, 10), '5 ~ 9'), (range(10), '0 ~ 9')):
                    v = [R[(model, c, m, s)] for s in seeds]
                    L.append(f"| {LABEL[m]} | {'총 공기만' if c == 'total' else '총 공기 + 배분'} | {lab} | {ms([x['air_m3'] for x in v], 0)} | {ms([x['nh4_cost'] for x in v])} | "
                             f"{ms([x['do_cost'] for x in v])} | {ms([x['cost'] for x in v])} | {ms([x['w_mean'] for x in v], 3)} |")
        L += ['', '| 방법 | 비교 | 둘 다 낮은 seed | 평균 공기 변화 | 평균 비용 변화 | 판정 |', '| --- | --- | --- | --- | --- | --- |']
        for m in ('ppo', 'fcsrl'):
            b5, m5, ok5 = judge(paired(R, model, m, range(5, 10)), 4)
            b10, m10, ok10 = judge(paired(R, model, m, range(10)), 8)
            tag = 'J1' if model == 'corrected' else 'J3'
            J[f'{tag}_{m}'] = dict(both_lower=b5, mean=m5, verdict='재현' if ok5 else '재현 실패')
            J[f'J2_{model}_{m}'] = dict(both_lower=b10, mean=m10, verdict='10 seed 이득 유지' if ok10 else '10 seed 기준 미달')
            L.append(f"| {LABEL[m]} | seed 5 ~ 9 | {b5}/5 | {m5['air_m3']:+,.0f} | {m5['cost']:+.3f} | {J[f'{tag}_{m}']['verdict']} |")
            L.append(f"| {LABEL[m]} | seed 0 ~ 9 | {b10}/10 | {m10['air_m3']:+,.0f} | {m10['cost']:+.3f} | {J[f'J2_{model}_{m}']['verdict']} |")
        L.append('')
    L += ['반복 사용한 개발 val 의 탐색이며 test 는 평가하지 않았다.']
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(dict(judgement=J, root=str(a.root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
