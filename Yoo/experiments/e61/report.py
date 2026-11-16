"""e61 집계: e60 과 같은 seed 짝 비교, Park 대비, 승수 동작."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E60 = [Path('/home/jio/fcsrl_src/results/e60/main'), Path('/home/jio/fcsrl_src/results/e60/main_rest')]
KEYS = ('air_m3', 'nh4_cost', 'do_cost', 'cost', 'nh4_excess_mg_h_L', 'do_excess_mg_h_L', 'worst_day_nh4_cost', 'days_within_limit_pct', 'intervention_pct')
LABEL = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL'}
PARK19, PARK22 = dict(air_m3=80748, cost=4.98), dict(air_m3=79633, cost=4.68)


def find(roots, m, s, name):
    hits = [r / 'runs' / m / f'seed_{s}' / name for r in roots if (r / 'runs' / m / f'seed_{s}' / name).exists()]
    assert len(hits) == 1, (m, s, name, hits)
    return json.loads(hits[0].read_text())


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def main(a):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    methods, seeds, limit = protocol['methods'], protocol['seeds'], protocol['train_cost_limit']
    new = {(m, s): find([a.root], m, s, 'metrics.json') for m in methods for s in seeds}
    old = {(m, s): find(E60, m, s, 'metrics.json') for m in methods for s in seeds}
    logs = {(m, s): find([a.root], m, s, 'train_log.json') for m in methods for s in seeds}
    cfg = {(m, s): find([a.root], m, s, 'effective_config.json') for m in methods for s in seeds}
    assert all(c['config']['cost_limit'] == limit for c in cfg.values())
    L = [f'# e61 결과: 학습 비용 한도 {limit} 로 승수를 묶은 학습 중 차단', '',
         f'e60 과 학습 PID 한도만 다르다 (12 에서 {limit}). e50 공통 v1.1 환경, 401,280전이, 마지막 정책, seed 5, 검증 40일 x 구성원 5개. 현장 또는 test 결과가 아니다.', '']
    J, summary = {}, {'means': {}, 'seed_level': {}}
    # J0
    L += ['## J0 승수가 실제로 묶였는가', '', '| 방법 | 승수 최댓값 > 0.05 seed | 승수 최댓값 | 마지막 승수 | 마지막 10갱신 학습 비용 | e60 같은 값 |', '| --- | --- | --- | --- | --- | --- |']
    for m in methods:
        mx = [max(r['lambda'] for r in logs[(m, s)]) for s in seeds]
        last = [logs[(m, s)][-1]['lambda'] for s in seeds]
        c10 = [np.mean([r['daily_cost_mean'] for r in logs[(m, s)][-10:]]) for s in seeds]
        c10_old = [np.mean([r['daily_cost_mean'] for r in find(E60, m, s, 'train_log.json')[-10:]]) for s in seeds]
        n = sum(x > 0.05 for x in mx)
        J[f'J0_{m}'] = dict(bound_seeds=n, lambda_max=mx, lambda_last=last, train_cost_last10=c10, e60_train_cost_last10=c10_old, binding=n >= 4)
        L.append(f'| {LABEL[m]} | {n}/5 | {ms(mx, 3)} | {ms(last, 3)} | {ms(c10)} | {ms(c10_old)} |')
    # 표
    head = ['| 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | NH4 초과 mg h/L | 최악 날짜 NH4 | 한도 12 날짜 % | 개입 % |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for cond in ('val_park_e19', 'val_park_e10', 'val_none', 'train_park_e10', 'train_none'):
        L += ['', f'## {cond}', ''] + head
        for m in methods:
            for tag, src in ((f'e61 한도 {limit}', new), ('e60 한도 12', old)):
                vals = [src[(m, s)][cond] for s in seeds]
                L.append(f"| {LABEL[m]} {tag} | {ms([v['air_m3'] for v in vals], 0)} | {ms([v['nh4_cost'] for v in vals])} | {ms([v['do_cost'] for v in vals])} | "
                         f"{ms([v['cost'] for v in vals])} | {ms([v['nh4_excess_mg_h_L'] for v in vals])} | {ms([v['worst_day_nh4_cost'] for v in vals], 1)} | "
                         f"{ms([v['days_within_limit_pct'] for v in vals], 1)} | {ms([v.get('intervention_pct', 0) for v in vals], 1)} |")
            vals = [new[(m, s)][cond] for s in seeds]
            summary['means'][f'{m}|{cond}'] = {k: float(np.mean([v.get(k, 0.0) for v in vals])) for k in KEYS}
            summary['seed_level'][f'{m}|{cond}'] = [{k: v.get(k, 0.0) for k in KEYS} for v in vals]
    # J1
    L += ['', '## J1 같은 seed 의 e61 minus e60 (val park_e19)', '', '| 방법 | seed | 공기 변화 | 비용 변화 | NH4 변화 | DO 변화 | 둘 다 감소 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for m in methods:
        rows = []
        for s in seeds:
            n_, o_ = new[(m, s)]['val_park_e19'], old[(m, s)]['val_park_e19']
            d = {k: n_[k] - o_[k] for k in ('air_m3', 'cost', 'nh4_cost', 'do_cost')}
            rows.append(d)
            L.append(f"| {LABEL[m]} | {s} | {d['air_m3']:+,.0f} | {d['cost']:+.3f} | {d['nh4_cost']:+.3f} | {d['do_cost']:+.3f} | {'예' if d['air_m3'] < 0 and d['cost'] < 0 else '아니오'} |")
        both = sum(r['air_m3'] < 0 and r['cost'] < 0 for r in rows)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        if not J[f'J0_{m}']['binding']:
            verdict = '판정 불가 (승수 미결속)'
        elif both >= 4 and mean['air_m3'] < 0 and mean['cost'] < 0:
            verdict = '개선'
        elif mean['cost'] < 0 and mean['air_m3'] > 0:
            verdict = '교환 (비용 감소, 공기 증가)'
        elif mean['air_m3'] < 0 and mean['cost'] > 0:
            verdict = '교환 (공기 감소, 비용 증가)'
        else:
            verdict = '개선 없음'
        J[f'J1_{m}'] = dict(paired=rows, mean=mean, both_lower_seeds=both, verdict=verdict)
        L.append(f"| {LABEL[m]} | 평균 | {mean['air_m3']:+,.0f} | {mean['cost']:+.3f} | {mean['nh4_cost']:+.3f} | {mean['do_cost']:+.3f} | {both}/5, {verdict} |")
    # J2, J3
    L += ['', '## J2, J3 Park 대비 (val park_e19, Park 값은 seed 평균이라 짝 비교 아님)', '',
          '| 방법 | 평균 공기 | 평균 비용 | Park e19 H1 보다 둘 다 낮은 seed | J2 | Park e22 H1 보다 두 평균 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    for m in methods:
        v = summary['means'][f'{m}|val_park_e19']
        n = sum(new[(m, s)]['val_park_e19']['air_m3'] < PARK19['air_m3'] and new[(m, s)]['val_park_e19']['cost'] < PARK19['cost'] for s in seeds)
        j2 = v['air_m3'] < PARK19['air_m3'] and v['cost'] < PARK19['cost'] and n >= 4
        j3 = v['air_m3'] < PARK22['air_m3'] and v['cost'] < PARK22['cost']
        J[f'J2_{m}'] = dict(seeds=n, candidate=bool(j2))
        J[f'J3_{m}'] = bool(j3)
        L.append(f"| {LABEL[m]} | {v['air_m3']:,.0f} | {v['cost']:.2f} | {n}/5 | {'탐색 후보' if j2 else '아님'} | {j3} |")
    # J4
    L += ['', '## J4 차단 의존 (val 차단 없음 비용)', '']
    for m in methods:
        L.append(f"- {LABEL[m]}: e61 {ms([new[(m, s)]['val_none']['cost'] for s in seeds])}, e60 {ms([old[(m, s)]['val_none']['cost'] for s in seeds])}")
    L += ['', '반복 사용한 개발 val 의 seed 5 탐색이며 통계적 확증이나 공식 채택을 주장하지 않는다.']
    summary.update(judgement=J, root=str(a.root))
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
