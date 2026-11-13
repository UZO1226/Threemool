"""e60 집계: 조건별 표, 사전 판정 J1~J5."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEYS = ('air_m3', 'nh4_cost', 'do_cost', 'cost', 'nh4_excess_mg_h_L', 'do_excess_mg_h_L', 'worst_day_nh4_cost', 'days_within_limit_pct',
        'intervention_pct', 'lowered_pct', 'policy_air_m3')
LABEL = {'ppo': 'PPO-Lag + 학습 중 차단', 'fcsrl': 'FCSRL + 학습 중 차단'}


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def row(label, vals):
    g = lambda k, nd=2: ms([v.get(k, 0.0) for v in vals], nd)  # noqa: E731
    return (f"| {label} | {g('air_m3', 0)} | {g('nh4_cost')} | {g('do_cost')} | {g('cost')} | {g('nh4_excess_mg_h_L')} | "
            f"{g('worst_day_nh4_cost', 1)} | {g('days_within_limit_pct', 1)} | {g('intervention_pct', 1)} | {g('lowered_pct', 1)} |")


def main(a):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    methods, seeds = protocol['methods'], protocol['seeds']
    M = {(m, s): json.loads((a.root / 'runs' / m / f'seed_{s}' / 'metrics.json').read_text()) for m in methods for s in seeds}
    e59 = json.loads((HERE.parent / 'e59' / 'summary.json').read_text())
    zero = e59['means']['zero|park_e19']
    park = protocol['references_same_environment']
    conds = [f'{sp}_{sh}' for sp, shs in protocol['eval'].items() for sh in shs]
    L = ['# e60 결과: 학습 중 최소 개입 차단을 켠 Yoo PPO-Lag / FCSRL', '',
         'e50 공통 v1.1 환경, 각 401,280전이, 마지막 정책, seed 5. 검증 40일 x 구성원 5개. 학습 중 차단은 park_e10 (1스텝, 다른 구성원, target 3.5).',
         '현장 또는 독립 test 결과가 아니다.', '']
    head = ['| 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | NH4 초과 mg h/L | 최악 날짜 NH4 | 한도 12 날짜 % | 개입 % | 내린 % |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    summary = {'means': {}, 'seed_level': {}}
    for cond in conds:
        L += [f'## {cond}', ''] + head
        for m in methods:
            vals = [M[(m, s)][cond] for s in seeds]
            L.append(row(LABEL[m], vals))
            summary['means'][f'{m}|{cond}'] = {k: float(np.mean([v.get(k, 0.0) for v in vals])) for k in KEYS}
            summary['seed_level'][f'{m}|{cond}'] = [{k: v.get(k, 0.0) for k in KEYS} for v in vals]
        L.append('')
    L += ['## seed 별 val park_e19 와 차단 없음', '', '| 방법 | seed | 공기 (e19) | 비용 (e19) | NH4 (e19) | DO (e19) | 비용 (없음) | 마지막 승수 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for m in methods:
        for s in seeds:
            v, n = M[(m, s)]['val_park_e19'], M[(m, s)]['val_none']
            L.append(f"| {m} | {s} | {v['air_m3']:,.0f} | {v['cost']:.2f} | {v['nh4_cost']:.2f} | {v['do_cost']:.2f} | {n['cost']:.2f} | {M[(m, s)]['final_lambda']:.3f} |")
    # J1
    p19 = park['park_e19_H1']
    L += ['', '## J1 Park e19 H1 (80,748 / 4.98, seed 평균) 대비', '', '| 방법 | 평균 공기 | 평균 비용 | 두 값 모두 낮은 seed | 판정 |', '| --- | --- | --- | --- | --- |']
    J = {}
    for m in methods:
        mean = summary['means'][f'{m}|val_park_e19']
        n = sum(M[(m, s)]['val_park_e19']['air_m3'] < p19['air_m3'] and M[(m, s)]['val_park_e19']['cost'] < p19['cost'] for s in seeds)
        ok = mean['air_m3'] < p19['air_m3'] and mean['cost'] < p19['cost'] and n >= 4
        J[f'J1_{m}'] = dict(seeds_both_lower=n, mean_both_lower=bool(mean['air_m3'] < p19['air_m3'] and mean['cost'] < p19['cost']), candidate=bool(ok))
        L.append(f"| {LABEL[m]} | {mean['air_m3']:,.0f} | {mean['cost']:.2f} | {n}/5 | {'탐색 후보' if ok else '아님'} |")
    # J2
    L += ['', f"## J2 e59 공기 0 정책 + park_e19 ({zero['air_m3']:,.0f} / {zero['cost']:.2f}) 대비 정책 기여", '', '| 방법 | 두 값 모두 낮은 seed | 판정 |', '| --- | --- | --- |']
    for m in methods:
        n = sum(M[(m, s)]['val_park_e19']['air_m3'] < zero['air_m3'] and M[(m, s)]['val_park_e19']['cost'] < zero['cost'] for s in seeds)
        J[f'J2_{m}'] = dict(seeds_both_lower=n, verdict='기여' if n >= 4 else ('기여 없음' if n == 0 else '불확정'))
        L.append(f"| {LABEL[m]} | {n}/5 | {J[f'J2_{m}']['verdict']} |")
    # J3
    L += ['', '## J3 같은 seed 에서 학습 중 차단의 효과 (e60 minus e59 같은 방법, 둘 다 park_e19 평가)', '', '| 방법 | 공기 변화 | 비용 변화 | 둘 다 감소 seed |', '| --- | --- | --- | --- |']
    for m in methods:
        prev = e59['seed_level'][f'{m}|park_e19']
        da = [M[(m, s)]['val_park_e19']['air_m3'] - prev[s]['air_m3'] for s in seeds]
        dc = [M[(m, s)]['val_park_e19']['cost'] - prev[s]['cost'] for s in seeds]
        n = sum(x < 0 and y < 0 for x, y in zip(da, dc))
        J[f'J3_{m}'] = dict(air_delta=da, cost_delta=dc, seeds_both_lower=n)
        L.append(f"| {LABEL[m]} | {ms(da, 0)} | {ms(dc)} | {n}/5 |")
    # J4, J5
    L += ['', '## J4 차단 의존', '']
    for m in methods:
        L.append(f"- {LABEL[m]}: 차단 없음 val 비용 {ms([M[(m, s)]['val_none']['cost'] for s in seeds])}, 마지막 승수 {ms([M[(m, s)]['final_lambda'] for s in seeds], 3)}")
    L += ['', '## J5 세 지표', '']
    for m in methods:
        v = summary['means'][f'{m}|val_park_e19']
        L.append(f"- {LABEL[m]} + park_e19: NH4 {v['nh4_cost']:.2f}, DO {v['do_cost']:.2f}, NH4 초과 {v['nh4_excess_mg_h_L']:.2f}, DO 초과 {v['do_excess_mg_h_L']:.3f}, 최악 날짜 NH4 {v['worst_day_nh4_cost']:.1f}")
    L += ['', '## 같은 환경 참고 (우열 아님)', '', '| 출처 | 공기량 | 합산 비용 |', '| --- | --- | --- |']
    for k, v in park.items():
        if isinstance(v, dict):
            L.append(f"| Park {k} | {v['air_m3']:,} | {v['cost']:.2f} |")
    L.append(f"| Yoo e59 공기 0 + park_e19 | {zero['air_m3']:,.0f} | {zero['cost']:.2f} |")
    L += ['', '반복 사용한 개발 val 의 seed 5 탐색이며 통계적 확증이나 공식 채택을 주장하지 않는다. Park 값은 seed 평균이라 짝 비교가 아니다.']
    summary.update(judgement=J, root=str(a.root))
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
