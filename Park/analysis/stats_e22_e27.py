"""Park/e22 ~ e27 seed 10개 짝 비교 재분석 (Park 작성). 결과를 본 뒤 하는 분석이라 판정을 바꾸지 않고 효과 크기만 잰다.

    python stats_e22_e27.py --res <결과>   (<결과> 아래 e19/seed_*.json, e22/ensemble.json, e23/e23.json, e24/e24.json, e25/e25.json)
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = list(range(10))
KEYS = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct')


def load(res):
    res = Path(res)
    single = {r['seed']: r['conf_l3.5_m0.1_rec'] for r in (json.loads(p.read_text()) for p in (res / 'e19').glob('seed_*.json'))}
    e23 = json.loads((res / 'e23' / 'e23.json').read_text())
    single.update({int(s): r for s, r in e23['single'].items()})
    e22 = json.loads((res / 'e22' / 'ensemble.json').read_text())
    ens = {}
    for c in itertools.combinations(range(5), 4):
        ens[c] = e22['4x16' if c == (0, 1, 2, 3) else '4x16_' + ''.join(map(str, c))]
    for c in itertools.combinations(range(5, 10), 4):
        ens[c] = e23['ensemble']['4x16_' + ''.join(map(str, c))]
    e24 = {int(s): r for s, r in json.loads((res / 'e24' / 'e24.json').read_text()).items()}
    e25 = json.loads((res / 'e25' / 'e25.json').read_text())
    return single, ens, e24, {h: {int(s): r for s, r in e25[h].items()} for h in ('H1', 'H2')}


def signflip_p(d):
    """짝 차이 부호를 모두 뒤집어 보는 정확 순열 검정 양측 p."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def summary(d):
    d = np.asarray(d, float)
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * sd / np.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h, p=signflip_p(d), neg=int((d < 0).sum()), mde=mde)


def row(name, key, s, unit_fmt):
    f = unit_fmt
    return (f"| {name} | {key} | {f(s['mean'])} | {f(s['lo'])} ~ {f(s['hi'])} | {s['neg']}/{s['n']} | {s['p']:.3f} | {f(s['mde'])} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--res', required=True)
    p.add_argument('--md')
    a = p.parse_args()
    single, ens, e24, e25 = load(a.res)
    fa = lambda v: f'{v:+,.0f}'  # noqa: E731
    fc = lambda v: f'{v:+.3f}'  # noqa: E731
    L = ['| 비교 | 지표 | 짝 차이 평균 | 95% 신뢰구간 (t) | 줄어든 짝 | 부호 순열 p | 80% 검출력 최소 효과 |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    comps = {
        'e24 학습률 6e-4 대 기준 (seed 짝 10)': [(e24[s], single[s]) for s in SEEDS],
        'e25 k = 3 대 기준 (seed 짝 10)': [(e25['H1'][s], single[s]) for s in SEEDS],
        'e25 k = 9 대 기준 (seed 짝 10)': [(e25['H2'][s], single[s]) for s in SEEDS],
    }
    out = {}
    for name, pairs in comps.items():
        for key, f in (('air_m3', fa), ('cost', fc)):
            s = summary([x[key] - y[key] for x, y in pairs])
            out[(name, key)] = s
            L.append(row(name, key, s, f))
    # 앙상블: 조합마다 앙상블 값 - 같은 4개 seed 정책 하나 값의 평균 (조합끼리 seed 를 나눠 가져서 서로 독립이 아니다)
    name = 'e22 4 x 16 앙상블 대 같은 seed 정책 하나 평균 (조합 10, 독립 아님)'
    for key, f in (('air_m3', fa), ('cost', fc)):
        d = [ens[c][key] - np.mean([single[s][key] for s in c]) for c in ens]
        s = summary(d)
        out[(name, key)] = s
        L.append(row(name, key, s, f))
    grp = []
    for g in ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)):
        cs = [c for c in ens if c[0] in g]
        grp.append((g, *[np.mean([ens[c][k] for c in cs]) - np.mean([single[s][k] for s in g]) for k in ('air_m3', 'cost')]))
    sd_single = {k: float(np.std([single[s][k] for s in SEEDS], ddof=1)) for k in ('air_m3', 'cost')}
    L += ['', '서로 겹치지 않는 seed 묶음 2개의 앙상블 효과 (조합 평균 - 정책 하나 평균):', '']
    for g, da, dc in grp:
        L.append(f'- seed {g[0]} ~ {g[-1]}: 공기량 {da:+,.0f}, 비용 {dc:+.3f}')
    L += ['', f"정책 하나 + e19 차단의 seed 사이 표준편차 (seed 10개): 공기량 {sd_single['air_m3']:,.0f}, 비용 {sd_single['cost']:.3f}"]
    text = '\n'.join(L)
    print(text)
    if a.md:
        Path(a.md).write_text(text + '\n')


if __name__ == '__main__':
    main()
