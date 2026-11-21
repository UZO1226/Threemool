"""e73 집계: 예산 속도 차단 아래의 e72 정책. 판정 S1 ~ S5 는 protocol.json. holdout, test 는 읽지 않는다."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABEL = {'fcsrl': 'FCSRL', 'ppolag': '고정 표현 PPO'}
KEYS = ('air_m3', 'over_budget_steps', 'day_violation_pct', 'nh4_excess_mg_h_L', 'nh4_mean', 'do_step_viol_pct')
D = ('air_m3', 'over_budget_steps', 'nh4_excess_mg_h_L', 'day_violation_pct')
E72 = Path('/home/jio/fcsrl_src/results/e72/main/runs')


def stoch(ev):
    runs = [r for r in ev['runs'] if r['mode'].startswith('stoch')]
    assert len(runs) == 3
    out = {k: float(np.mean([r[k] for r in runs])) for k in KEYS}
    out.update({k: float(np.mean([r['shield'][k] for r in runs])) for k in runs[0]['shield']})
    return out


def det(ev):
    r = [r for r in ev['runs'] if r['mode'] == 'det'][0]
    return dict({k: r[k] for k in KEYS}, **r['shield'])


def ms(x, nd=2):
    x = np.asarray(x, float)
    return f'{x.mean():,.{nd}f} ± {x.std():,.{nd}f}'


def main(a):
    root, n = a.root, a.seeds
    man = json.loads((root / 'manifest.json').read_text())
    B = json.loads(Path('/home/jio/fcsrl_src/results/e68/main/baselines/baselines.json').read_text())['chosen']
    ev = lambda name: json.loads((root / 'evals' / f'{name}.json').read_text())  # noqa: E731
    S = {sp: {m: [stoch(ev(f'{m}_s{s}_{sp}')) for s in range(n)] for m in ('fcsrl', 'ppolag')} for sp in ('dev', 'val')}
    Dt = {sp: {m: [det(ev(f'{m}_s{s}_{sp}')) for s in range(n)] for m in ('fcsrl', 'ppolag')} for sp in ('dev', 'val')}
    A0 = {sp: det(ev(f'air0_{sp}')) for sp in ('dev', 'val')}
    NS = {sp: {m: [json.loads((E72 / f'{m}_s{s}' / 'metrics.json').read_text())[f'{sp}_stochastic'] for s in range(n)] for m in ('fcsrl', 'ppolag')} for sp in ('dev', 'val')}
    need = int(np.ceil(0.8 * n))
    better = lambda d: d['air_m3'] < 0 and d['over_budget_steps'] <= 0 and d['nh4_excess_mg_h_L'] <= 0  # noqa: E731
    worse = lambda d: d['air_m3'] > 0 and d['over_budget_steps'] >= 0 and d['nh4_excess_mg_h_L'] >= 0  # noqa: E731
    L = [f"# e73 결과: e72 정책 + 예산 속도 차단 (margin {man['margin']:g})", '',
         '대리 환경 결과이며 현장 성능이 아니다. dev 146일 x 구성원 5 (주), val 40일 x 5 (보조). 정책은 표본 행동 (고정 seed 3개 평균) 이 주 평가. 새 학습 없음. holdout, test 미사용.', '']
    head = '| 방법 | 공기량 m³/23.75h | 예산 초과 스텝/일 | 하루 위반율 % | NH4 초과 mg h/L | NH4 평균 | DO>2 스텝 % | 차단 개입 % | 해 없음 % | 0.3 이하 seed |'
    for sp in ('dev', 'val'):
        L += [f'## {sp}', '', head, '|' + ' --- |' * 10]
        for m in ('fcsrl', 'ppolag'):
            rows = S[sp][m]
            L.append(f'| {LABEL[m]} + 차단 | ' + ' | '.join(ms([r[k] for r in rows], 0 if k == 'air_m3' else 2) for k in KEYS)
                     + f" | {ms([r['intervention_pct'] for r in rows], 1)} | {ms([r['infeasible_pct'] for r in rows], 2)} | {sum(r['over_budget_steps'] <= 0.3 for r in rows)}/{n} |")
            rows = Dt[sp][m]
            L.append(f'| {LABEL[m]} + 차단, 평균 행동 (참고) | ' + ' | '.join(ms([r[k] for r in rows], 0 if k == 'air_m3' else 2) for k in KEYS)
                     + f" | {ms([r['intervention_pct'] for r in rows], 1)} | {ms([r['infeasible_pct'] for r in rows], 2)} | {sum(r['over_budget_steps'] <= 0.3 for r in rows)}/{n} |")
            rows = NS[sp][m]
            L.append(f'| {LABEL[m]} 차단 없음 (e72) | ' + ' | '.join(ms([r[k] for r in rows], 0 if k == 'air_m3' else 2) for k in KEYS) + f" | | | {sum(r['over_budget_steps'] <= 0.3 for r in rows)}/{n} |")
        a0 = A0[sp]
        L.append('| 공기 0 + 차단 (차단만의 제어기) | ' + ' | '.join(f"{a0[k]:,.0f}" if k == 'air_m3' else f'{a0[k]:.2f}' for k in KEYS) + f" | {a0['intervention_pct']:.1f} | {a0['infeasible_pct']:.2f} | |")
        ab = B['abac'][sp]
        L.append('| ABAC (설정값 2.9) | ' + ' | '.join(f"{ab[k]:,.0f}" if k == 'air_m3' else f'{ab[k]:.2f}' for k in KEYS) + ' | | | |')
        L.append('')
    J = dict(margin=man['margin'])
    L += ['## 판정 (사전 규칙)', '']
    J['S1'] = {sp: {m: sum(r['over_budget_steps'] <= 0.3 for r in S[sp][m]) for m in S[sp]} for sp in ('dev', 'val')}
    for m in ('fcsrl', 'ppolag'):
        k = J['S1']['dev'][m]
        L.append(f"- S1 {LABEL[m]} + 차단: dev 예산 초과 0.3 이하 seed {k}/{n} -> **{'한도 충족' if k >= need else '한도 미달'}** (val 참고 {J['S1']['val'][m]}/{n})")
    for sp in ('dev', 'val'):
        pairs = [{k: S[sp]['fcsrl'][s][k] - S[sp]['ppolag'][s][k] for k in D} for s in range(n)]
        f, p = sum(map(better, pairs)), sum(map(worse, pairs))
        v = 'FCSRL 우위' if f >= need else '고정 표현 PPO 우위' if p >= need else '차이 불명확'
        per = {k: sum(x[k] < 0 for x in pairs) for k in D}
        J[f'S2_{sp}'] = dict(pairs=pairs, fcsrl_better=f, ppolag_better=p, verdict=v, fcsrl_lower_by_metric=per)
        L.append(f"- S2 {sp}{' (주)' if sp == 'dev' else ' (보조)'}: FCSRL 우위 짝 {f}/{n}, 고정 표현 PPO 우위 짝 {p}/{n} -> **{v}**. 지표별 FCSRL 이 낮은 짝: "
                 + ', '.join(f'{k} {v_}/{n}' for k, v_ in per.items()) + '. 평균 차이: ' + ', '.join(f"{k} {np.mean([x[k] for x in pairs]):+,.3f}" for k in D))
    lower = sum(S['dev']['fcsrl'][s]['intervention_pct'] < S['dev']['ppolag'][s]['intervention_pct'] for s in range(n))
    J['S3'] = dict(fcsrl_lower=lower, fcsrl=[r['intervention_pct'] for r in S['dev']['fcsrl']], ppolag=[r['intervention_pct'] for r in S['dev']['ppolag']])
    L.append(f"- S3 dev 차단 개입률: FCSRL 이 낮은 짝 {lower}/{n} (평균 {np.mean(J['S3']['fcsrl']):.1f} % 대 {np.mean(J['S3']['ppolag']):.1f} %) -> **{'FCSRL 이 차단을 덜 부름' if lower >= need else '고정 표현 PPO 가 덜 부름' if n - lower >= need else '차이 불명확'}**")
    J['S4'] = {}
    for m in ('fcsrl', 'ppolag'):
        k = sum(better({x: r[x] - A0['dev'][x] for x in D}) for r in S['dev'][m])
        J['S4'][m] = k
        L.append(f"- S4 {LABEL[m]} + 차단 대 공기 0 + 차단 (dev): 공기 낮고 예산 초과, NH4 초과가 같거나 낮은 seed {k}/{n} -> **{'정책 기여 있음' if k >= need else '정책 기여 없음' if k == 0 else '불확정'}**")
    ab = B['abac']['dev']
    better3 = lambda d: d['air_m3'] < 0 and d['over_budget_steps'] <= 0 and d['day_violation_pct'] <= 0 and d['nh4_excess_mg_h_L'] <= 0  # noqa: E731
    J['S5'] = {}
    for m in ('fcsrl', 'ppolag'):
        k = sum(better3({x: r[x] - ab[x] for x in D}) for r in S['dev'][m])
        J['S5'][m] = k
        L.append(f"- S5 {LABEL[m]} + 차단 대 ABAC (dev): 네 지표 모두 나은 seed {k}/{n}. 평균 공기 {np.mean([r['air_m3'] for r in S['dev'][m]]):,.0f} vs {ab['air_m3']:,.0f}, "
                 f"위반일 {np.mean([r['day_violation_pct'] for r in S['dev'][m]]):.2f} vs {ab['day_violation_pct']:.2f}, NH4 초과 {np.mean([r['nh4_excess_mg_h_L'] for r in S['dev'][m]]):.2f} vs {ab['nh4_excess_mg_h_L']:.2f}")
    J['S5']['air0'] = bool(better3({x: A0['dev'][x] - ab[x] for x in D}))
    L.append(f"- S5 공기 0 + 차단 대 ABAC (dev): {'네 지표 모두 나음' if J['S5']['air0'] else '우위 아님'}")
    L += ['', '## seed 별 (dev, 표본 행동)', '', '| 방법 | seed | 공기 | 예산 초과 | NH4 초과 | 차단 개입 % | 차단 없음 예산 초과 (e72) |', '| --- | --- | --- | --- | --- | --- | --- |']
    for m in ('fcsrl', 'ppolag'):
        for s in range(n):
            r = S['dev'][m][s]
            L.append(f"| {LABEL[m]} | {s} | {r['air_m3']:,.0f} | {r['over_budget_steps']:.3f} | {r['nh4_excess_mg_h_L']:.2f} | {r['intervention_pct']:.1f} | {NS['dev'][m][s]['over_budget_steps']:.3f} |")
    L += ['', 'holdout 과 test 는 이 실험에서 쓰지 않았다.']
    (HERE / 'results.md').write_text('\n'.join(L) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(dict(judgement=J, root=str(root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=10)
    main(p.parse_args())
