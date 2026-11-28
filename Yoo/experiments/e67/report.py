"""e67 집계: J1 (규칙 + 차단 대비), J2 (총 공기 정책 + 고정 배분 대비)."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEARNED = {s: (Path('/home/jio/fcsrl_src/results/e64/main/runs') if s < 5 else Path('/home/jio/fcsrl_src/results/e65/main/eval')) for s in range(10)}


def main(a):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    W = protocol['fixed_w']
    L = ['# e67 결과: 학습된 배분 대 고정 배분', '', '검증 40일 x 구성원 5개, 일별, 실행 상한, park_e19. 새 학습 없음, test 없음.', '']
    J = {}
    for model, title in (('corrected', 'DO 수정 모델 (주)'), ('common', '공통 v1.1 모델')):
        get = lambda name: json.loads((a.root / 'runs' / model / name / 'result.json').read_text())['val']  # noqa: E731
        learned = [json.loads((LEARNED[s] / f'{model}_cap1_daily' / f'alloc_ppo_s{s}' / 'result.json').read_text())['val'] for s in range(10)]
        total0 = [json.loads((LEARNED[s] / f'{model}_cap1_daily' / f'total_ppo_s{s}' / 'result.json').read_text())['val'] for s in range(10)]
        L += [f'## {title}', '', '| 제어기 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 |', '| --- | --- | --- | --- | --- |']
        f = lambda v, k: np.mean([x[k] for x in v])  # noqa: E731
        L.append(f"| 학습된 배분 PPO-Lag (seed 10) | {f(learned, 'air_m3'):,.0f} | {f(learned, 'nh4_cost'):.2f} | {f(learned, 'do_cost'):.2f} | {f(learned, 'cost'):.2f} |")
        L.append(f"| 총 공기만 PPO-Lag, 기록 배분 (seed 10) | {f(total0, 'air_m3'):,.0f} | {f(total0, 'nh4_cost'):.2f} | {f(total0, 'do_cost'):.2f} | {f(total0, 'cost'):.2f} |")
        zero = {w: get(f'zero_w{w}') for w in W}
        pt = {w: [get(f'ppo_total_s{s}_w{w}') for s in range(10)] for w in W}
        for w in W:
            L.append(f"| 총 공기만 PPO-Lag + 고정 w {w} (seed 10) | {f(pt[w], 'air_m3'):,.0f} | {f(pt[w], 'nh4_cost'):.2f} | {f(pt[w], 'do_cost'):.2f} | {f(pt[w], 'cost'):.2f} |")
        for w in W:
            z = zero[w]
            L.append(f"| 공기 0 + 고정 w {w} + 차단 | {z['air_m3']:,.0f} | {z['nh4_cost']:.2f} | {z['do_cost']:.2f} | {z['cost']:.2f} |")
        best = min(W, key=lambda w: zero[w]['cost'])
        n1 = sum(x['air_m3'] < zero[best]['air_m3'] and x['cost'] < zero[best]['cost'] for x in learned)
        n2 = sum(x['air_m3'] < y['air_m3'] and x['cost'] < y['cost'] for x, y in zip(learned, pt[0.13]))
        others = {w: sum(x['air_m3'] < y['air_m3'] and x['cost'] < y['cost'] for x, y in zip(learned, pt[w])) for w in W}
        J[model] = dict(best_rule_w=best, J1_seeds=n1, J1='RL 배분이 규칙 + 차단보다 낫다' if n1 >= 8 else '규칙 + 차단과 구별되지 않음',
                        J2_seeds=n2, J2='상태 의존 배분이 고정 배분보다 낫다' if n2 >= 8 else '고정 배분과 구별되지 않음', J2_all_w=others)
        L += ['', f"- J1: 규칙 + 차단 중 비용 최저는 w {best} ({zero[best]['air_m3']:,.0f} / {zero[best]['cost']:.2f}). 학습된 배분 정책이 공기와 비용 모두 낮은 seed {n1}/10. 판정: {J[model]['J1']}.",
              f"- J2: 같은 seed 의 총 공기만 정책 + 고정 w 0.13 대비 공기와 비용 모두 낮은 seed {n2}/10. 판정: {J[model]['J2']}. (w 별: " + ', '.join(f'{w}: {n}/10' for w, n in others.items()) + ')', '']
    L += ['반복 사용한 개발 val 의 재평가다.']
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(J, indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
