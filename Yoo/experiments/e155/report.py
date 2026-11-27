"""e155 집계: 사전 검사, 조건별 평균 표, H1 ~ H4 판정. README 의 규칙을 그대로 따른다.

    report.py --out <결과> --md results.md --json summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import run as R

SPLITS = (('train', 'v11'), ('val', 'v11'), ('val', 'dofix'), ('val', 'past148'))
CTRLS = ('abac',) + R.METHODS
LABEL = {'abac': 'ABAC', 'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL', 'fcsrl_actor': 'FCSRL + Actor 보정'}
MOD = {'none': '없음', 'shield': '차단', 'risk': '위험 선택'}


def job(split, ev, ctrl, seed, module, scale=1000):
    return dict(split=split, ev=ev, ctrl=ctrl, seed=seed, module=module, scale=scale)


def seeds_of(ctrl):
    return (0,) if ctrl == 'abac' else R.SEEDS


def res(out, j):
    return json.loads((out / f'{R.name(j)}.json').read_text())['result']


def logs(out, split, ev, ctrl, module, scale=1000):
    return [dict(np.load(out / f'log_{R.name(job(split, ev, ctrl, s, module, scale))}.npz')) for s in seeds_of(ctrl)]


def daily(out, split, ev, ctrl, module):
    gs = logs(out, split, ev, ctrl, module)
    days, c = R.series(gs, 'cost')
    _, a = R.series(gs, 'air')
    return days, c, a


def compare(out, split, ev, ctrl, mod_a, mod_b):
    """(mod_a 빼기 mod_b) 날짜별 비용과 공기량 차이, 블록 구간 (비교마다 seed 0)."""
    da, ca, aa = daily(out, split, ev, ctrl, mod_a)
    db, cb, ab = daily(out, split, ev, ctrl, mod_b)
    assert da == db
    dc, dair = ca - cb, aa - ab
    clo, chi = R.block_ci(dc, np.random.default_rng(R_SEED))
    alo, ahi = R.block_ci(dair, np.random.default_rng(R_SEED))
    return dict(n=len(dc), cost=float(dc.mean()), cost_ci=(float(clo), float(chi)), air=float(dair.mean()), air_ci=(float(alo), float(ahi)),
                better=int((dc < 0).sum()), same=int((dc == 0).sum()), worse=int((dc > 0).sum()))


R_SEED = 0


def fmt(c):
    return (f"{c['cost']:+.3f} | {c['cost_ci'][0]:+.3f} ~ {c['cost_ci'][1]:+.3f} | {c['air']:+,.0f} | "
            f"{c['air_ci'][0]:+,.0f} ~ {c['air_ci'][1]:+,.0f} | {c['better']} / {c['same']} / {c['worse']}")


CMP_HEAD = ['| 비교 | 날 수 | 하루 비용 차이 | 블록 95% 구간 | 공기량 차이 | 공기량 블록 95% 구간 | 나은 날 / 같은 날 / 나쁜 날 |',
            '| --- | --- | --- | --- | --- | --- | --- |']


def three_way(c):
    lo, hi = c['cost_ci']
    if hi < 0:
        return '아래'
    if lo > 0:
        return '위'
    return '0 포함'


def main(a):
    out = a.out
    S, L = {}, []
    # ---- 사전 검사 ----
    L += ['# e155 결과', '', '대리 환경 결과이며 현장 성능이 아니다. test 41일은 불러오지 않았다. 판정 규칙은 [README](README.md).', '', '## 사전 검사', '']
    rep = {f'{m}_s{s}': res(out, job('val', 'v11', m, s, 'none')).get('reproduced_e50') for m in R.METHODS for s in R.SEEDS}
    S['reproduced_e50'] = rep
    L.append(f"- e50 검증 지표 재현 (차단 없음, v1.1): {sum(bool(v) for v in rep.values())}/{len(rep)}")
    park = {}
    for (split, module), (air, cost) in R.PARK_E86.items():
        r = res(out, job(split, 'v11', 'abac', 0, module))
        ok = round(r['air_m3']) == air and round(r['cost'], 3) == cost
        park[f'{split}_{module}'] = dict(ours=(r['air_m3'], r['cost']), park=(air, cost), same=ok)
        L.append(f"- Park/e86 ABAC {split} {MOD[module]}: {r['air_m3']:,.1f} / {r['cost']:.4f} 대 Park {air:,} / {cost} -> {'같음' if ok else '다름'}")
    S['park_e86'] = park
    fit = json.loads((out / 'risk_fit.json').read_text())
    L.append(f"- 위험 선택기 적합 행 수 {fit[0]['rows']:,} (seed 마다), 빠른 예측과 sklearn 차이 1e-12 이내 (적합 중 검사)")

    # ---- 조건별 평균 ----
    L += ['', '## 조건별 평균 (seed 평균, 구성원 5개)', '',
          '| 날 | 평가 모델 | 제어기 | 모듈 | 공기량 m³/23.75h | 하루 비용 | NH4 위반 % | DO 위반 % | 개입 % | 내린 스텝 % |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    S['means'] = {}
    for split, ev in SPLITS:
        for c in CTRLS:
            for m in ('none', 'shield', 'risk'):
                if c == 'abac' and m == 'risk':
                    continue
                rs = [res(out, job(split, ev, c, s, m)) for s in seeds_of(c)]
                mean = {k: float(np.mean([r.get(k, 0.0) for r in rs])) for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'intervention_pct')}
                mean['lowered_pct'] = float(np.mean([100 * r['lowered_steps'] / r['steps'] for r in rs])) if m == 'shield' else 0.0
                S['means'][f'{split}_{ev}_{c}_{m}'] = mean
                L.append(f"| {split} | {ev} | {LABEL[c]} | {MOD[m]} | {mean['air_m3']:,.0f} | {mean['cost']:.3f} | {mean['nh4_viol_pct']:.2f} | "
                         f"{mean['do_viol_pct']:.2f} | {mean['intervention_pct']:.2f} | {mean['lowered_pct']:.2f} |")

    # ---- H1 ----
    L += ['', '## H1 차단의 몫 (차단 빼기 없음, v1.1)', ''] + CMP_HEAD
    H1 = {}
    for split in ('train', 'val'):
        for c in CTRLS:
            cmp_ = compare(out, split, 'v11', c, 'shield', 'none')
            H1[f'{split}_{c}'] = cmp_
            L.append(f"| {split} {LABEL[c]} | {cmp_['n']} | {fmt(cmp_)} |")
    h1 = all(H1[f'train_{m}']['cost_ci'][1] < 0 for m in R.METHODS)
    S['H1'] = dict(compare=H1, supported=h1)
    L += ['', f"- H1 (학습 188일, 세 방법 모두 구간 0 아래): **{'지지' if h1 else '미지지'}**. "
          + ', '.join(f"{LABEL[m]} {three_way(H1[f'train_{m}'])}" for m in R.METHODS)
          + f". 검증 확인: " + ', '.join(f"{LABEL[m]} {three_way(H1[f'val_{m}'])}" for m in R.METHODS)]

    # ---- H2 ----
    L += ['', '## H2 같은 실행 공기량에서의 차단 개입 (검증, v1.1)', '',
          '| seed | FCSRL 공기 | FCSRL 개입 % | PPO-Lag 세 점 공기 범위 | 보간 PPO-Lag 개입 % | 개입 차이 | 보간 비용 차이 (참고) | 반대 방향 개입 차이 (참고) |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    H2 = []
    for s in R.SEEDS:
        def pts(m):
            P = sorted((res(out, job('val', 'v11', m, s, 'shield', k)) for k in (900, 1000, 1100)), key=lambda r: r['air_m3'])
            return np.array([r['air_m3'] for r in P]), np.array([r['intervention_pct'] for r in P]), np.array([r['cost'] for r in P])
        pa, pi, pc = pts('ppo')
        fa, fi, fc = pts('fcsrl')
        f1 = res(out, job('val', 'v11', 'fcsrl', s, 'shield'))
        p1 = res(out, job('val', 'v11', 'ppo', s, 'shield'))
        inside = pa[0] <= f1['air_m3'] <= pa[-1]
        d = float(f1['intervention_pct'] - np.interp(f1['air_m3'], pa, pi)) if inside else None
        dc = float(f1['cost'] - np.interp(f1['air_m3'], pa, pc)) if inside else None
        rev_in = fa[0] <= p1['air_m3'] <= fa[-1]
        drev = float(np.interp(p1['air_m3'], fa, fi) - p1['intervention_pct']) if rev_in else None
        H2.append(dict(seed=s, fcsrl_air=f1['air_m3'], fcsrl_int=f1['intervention_pct'], ppo_range=(float(pa[0]), float(pa[-1])),
                       ppo_int_interp=None if d is None else f1['intervention_pct'] - d, diff=d, cost_diff=dc, reverse_diff=drev))
        g = lambda v: '범위 밖' if v is None else f'{v:+.3f}'  # noqa: E731
        interp = '범위 밖' if d is None else f"{f1['intervention_pct'] - d:.3f}"
        L.append(f"| {s} | {f1['air_m3']:,.0f} | {f1['intervention_pct']:.3f} | {pa[0]:,.0f} ~ {pa[-1]:,.0f} | "
                 f"{interp} | {g(d)} | {g(dc)} | {g(drev)} |")
    neg = sum(1 for h in H2 if h['diff'] is not None and h['diff'] < 0)
    vals = [h['diff'] for h in H2 if h['diff'] is not None]
    h2 = neg >= 4 and len(vals) > 0 and float(np.mean(vals)) < 0
    S['H2'] = dict(rows=H2, negative=neg, mean=(float(np.mean(vals)) if vals else None), supported=h2)
    L += ['', f"- H2: FCSRL 개입이 낮은 seed {neg}/5 (범위 밖은 미지지), 평균 차이 {S['H2']['mean'] if vals else '없음'} -> **{'지지' if h2 else '미지지'}**"]

    # ---- H3 ----
    L += ['', '## H3 위험 선택 빼기 차단 (같은 정책)', ''] + CMP_HEAD
    H3 = {}
    for split in ('val', 'train'):
        for m in R.METHODS:
            c = compare(out, split, 'v11', m, 'risk', 'shield')
            lo, hi = c['cost_ci']
            alo, ahi = c['air_ci']
            if lo > 0 and not ahi < 0:
                v = '차단 우위'
            elif hi < 0 and not alo > 0:
                v = '위험 선택 우위'
            else:
                v = '가르지 못함'
            c['verdict'] = v
            H3[f'{split}_{m}'] = c
            L.append(f"| {split}{' (참고)' if split == 'train' else ''} {LABEL[m]} | {c['n']} | {fmt(c)} |")
    S['H3'] = H3
    L += ['', '- H3 (검증): ' + ', '.join(f"{LABEL[m]} **{H3[f'val_{m}']['verdict']}**" for m in R.METHODS)
          + '. 학습 188일 (참고, 위험 선택기를 그 날로 적합): ' + ', '.join(f"{LABEL[m]} {H3[f'train_{m}']['verdict']}" for m in R.METHODS)]

    # ---- H4 ----
    L += ['', '## H4 다른 대리 모델에서의 차단과 위험 선택 (검증)', ''] + CMP_HEAD
    H4 = {}
    for ev in ('past148', 'dofix'):
        for c in CTRLS:
            for m in ('shield', 'risk'):
                if c == 'abac' and m == 'risk':
                    continue
                cmp_ = compare(out, 'val', ev, c, m, 'none')
                H4[f'{ev}_{c}_{m}'] = cmp_
                L.append(f"| {ev} {LABEL[c]} {MOD[m]} 빼기 없음 | {cmp_['n']} | {fmt(cmp_)} |")
    h4 = all(H4[f'past148_{c}_shield']['cost_ci'][1] < 0 for c in CTRLS)
    S['H4'] = dict(compare=H4, supported=h4)
    L += ['', f"- H4 (past148, ABAC 와 세 방법 차단 모두 구간 0 아래): **{'지지' if h4 else '미지지'}**. "
          + ', '.join(f"{LABEL[c]} {three_way(H4[f'past148_{c}_shield'])}" for c in CTRLS)
          + '. dofix 차단: ' + ', '.join(f"{LABEL[c]} {three_way(H4[f'dofix_{c}_shield'])}" for c in CTRLS)]
    S['test_evaluated'] = False
    Path(a.md).write_text('\n'.join(L) + '\n')
    Path(a.json).write_text(json.dumps(S, indent=1, ensure_ascii=False) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--md', default='results.md')
    p.add_argument('--json', default='summary.json')
    main(p.parse_args())
