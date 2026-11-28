"""e69 집계: e68 집계와 같은 판정 (H1 ~ H3) 에 e68 (갱신 250번) 대비 변화와 재현 검사를 더한다. holdout, test 는 읽지 않는다."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABEL = {'fcsrl': 'FCSRL', 'ppolag': 'PPO-Lag (같은 망, 표현 학습 없음)'}
KEYS = ('air_m3', 'air_m3_ok_days', 'over_budget_steps', 'day_violation_pct', 'nh4_excess_mg_h_L', 'nh4_mean', 'nh4_step_viol_pct', 'nh4_day_mean_p90', 'do_step_viol_pct', 'no3_mean')


def ms(x, nd=2):
    x = np.asarray(x, float)
    return f'{x.mean():,.{nd}f} ± {x.std():,.{nd}f}'


def main(a):
    root = a.root
    B = json.loads(Path('/home/jio/fcsrl_src/results/e68/main/baselines/baselines.json').read_text())['chosen']
    E68 = Path('/home/jio/fcsrl_src/results/e68/main/runs')
    R = {}
    for m in ('fcsrl', 'ppolag'):
        R[m] = [json.loads((root / 'runs' / f'{m}_s{s}' / 'metrics.json').read_text()) for s in range(a.seeds)]
    logs = {m: [json.loads((root / 'runs' / f'{m}_s{s}' / 'train_log.json').read_text()) for s in range(a.seeds)] for m in R}
    L = ['# e69 결과: e68 과 같은 설정, 갱신 500번', '',
         '대리 환경 결과이며 현장 성능이 아니다. val 40일 x 구성원 5개, dev 146일 x 5. RL 은 표본 행동 (고정 seed 3개 평균) 이 주 평가. holdout, test 미사용.', '']
    head = '| 방법 | 공기량 m³/23.75h | 위반 없는 날 공기 | 예산 초과 스텝/일 | 하루 위반율 % | NH4 초과 mg h/L | NH4 평균 | NH4 스텝 위반 % | 하루 평균 NH4 p90 | DO>2 스텝 % | NO3 평균 |'
    for split in ('val', 'dev'):
        L += [f'## {split}', '', head, '|' + ' --- |' * (len(KEYS) + 1)]
        for m, rows in R.items():
            L.append(f'| {LABEL[m]} | ' + ' | '.join(ms([np.nan if r[split + "_stochastic"][k] is None else r[split + "_stochastic"][k] for r in rows], 0 if k.startswith('air') else 2) for k in KEYS) + ' |')
            L.append(f'| {LABEL[m]} 평균 행동 (참고) | ' + ' | '.join(ms([np.nan if r[split][k] is None else r[split][k] for r in rows], 0 if k.startswith('air') else 2) for k in KEYS) + ' |')
        for k, lab in (('record', '기록 운전'), ('constant', f"고정 공기 {B['constant']['params'].get('air', 0):.0f}"),
                       ('abac', 'ABAC ' + ', '.join(f'{x}={v}' for x, v in B['abac']['params'].items()))):
            L.append(f'| {lab} | ' + ' | '.join('없음' if B[k][split][x] is None else f"{B[k][split][x]:,.0f}" if x.startswith('air') else f"{B[k][split][x]:.2f}" for x in KEYS) + ' |')
        L.append('')
    J = {}
    need = int(np.ceil(0.8 * a.seeds))
    better = lambda d: d['air_m3'] < 0 and d['over_budget_steps'] <= 0 and d['nh4_excess_mg_h_L'] <= 0  # noqa: E731
    worse = lambda d: d['air_m3'] > 0 and d['over_budget_steps'] >= 0 and d['nh4_excess_mg_h_L'] >= 0  # noqa: E731
    D = ('air_m3', 'over_budget_steps', 'nh4_excess_mg_h_L', 'day_violation_pct')
    L += ['## 판정 (사전 규칙)', '']
    for split in ('dev', 'val'):
        sk = split + '_stochastic'
        pairs = [{k: R['fcsrl'][s][sk][k] - R['ppolag'][s][sk][k] for k in D} for s in range(a.seeds)]
        f_win, p_win = sum(map(better, pairs)), sum(map(worse, pairs))
        v = 'FCSRL 우위' if f_win >= need else 'PPO-Lag 우위' if p_win >= need else '차이 불명확'
        J[f'H1_{split}'] = dict(pairs=pairs, fcsrl_better=f_win, ppolag_better=p_win, verdict=v)
        L.append(f"- H1 {split}{' (주)' if split == 'dev' else ' (보조)'}: FCSRL 우위 짝 {f_win}/{a.seeds}, PPO-Lag 우위 짝 {p_win}/{a.seeds} -> **{v}**. 평균 차이 (FCSRL - PPO-Lag): "
                 + ', '.join(f"{k} {np.mean([p[k] for p in pairs]):+,.2f}" for k in D))
    rec_val = B['record']['val']['day_violation_pct']
    J['H2'] = {m: dict(dev_le_limit=sum(r['dev_stochastic']['over_budget_steps'] <= 0.3 for r in rows),
                       val_le_record=sum(r['val_stochastic']['day_violation_pct'] <= rec_val for r in rows)) for m, rows in R.items()}
    for m in R:
        L.append(f"- H2 {LABEL[m]}: dev 예산 초과 스텝 0.3 이하 seed {J['H2'][m]['dev_le_limit']}/{a.seeds}, val 위반율이 기록 운전 ({rec_val:.2f} %) 이하 seed {J['H2'][m]['val_le_record']}/{a.seeds}")
    for split in ('dev', 'val'):
        ab = B['abac'][split]
        for m, rows in R.items():
            sk = split + '_stochastic'
            mean = {k: float(np.mean([r[sk][k] for r in rows])) for k in D}
            d = {k: mean[k] - ab[k] for k in D}
            n = sum(better({k: r[sk][k] - ab[k] for k in D}) for r in rows)
            v = 'ABAC 대비 우위' if better(d) else '우위 아님'
            J[f'H3_{split}_{m}'] = dict(mean=mean, abac={k: ab[k] for k in D}, seeds_better=n, verdict=v)
            L.append(f"- H3 {split} {LABEL[m]}: 공기 {mean['air_m3']:,.0f} vs ABAC {ab['air_m3']:,.0f}, 예산 초과 스텝 {mean['over_budget_steps']:.3f} vs {ab['over_budget_steps']:.3f}, 위반율 {mean['day_violation_pct']:.2f} vs {ab['day_violation_pct']:.2f}, "
                     f"NH4 초과 {mean['nh4_excess_mg_h_L']:.2f} vs {ab['nh4_excess_mg_h_L']:.2f} -> {v} (세 지표 모두 나은 seed {n}/{a.seeds})")
    L += ['', '## 학습 곡선 (학습 수집 기준, seed 평균)', '', '| 방법 | 구간 | 예산 초과 스텝/일 | 공기 | 승수 | 표준편차 | 가능성 목표 > 0.05 비율 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    for m, ls in logs.items():
        n = len(ls[0])
        for lo, hi in ((0, 25), (n // 2 - 12, n // 2 + 13), (n - 50, n - 25), (n - 25, n)):
            seg = [r for log in ls for r in log[lo:hi]]
            L.append(f"| {LABEL[m]} | 갱신 {lo + 1} ~ {hi} | {np.mean([r['cost'] for r in seg]):.2f} | {np.mean([r['air_m3'] for r in seg]):,.0f} | {np.mean([r['lagrg'] for r in seg]):.3f} | {np.mean([r['sigma'] for r in seg]):.3f} | {np.mean([r['feasi_target_gt05_pct'] for r in seg]):.2f} |")
    L += ['', '## e68 (갱신 250번) 대비, dev 표본 행동, seed 평균', '', '| 방법 | 갱신 | 공기 | 예산 초과 스텝/일 | 하루 위반율 % | NH4 초과 |', '| --- | --- | --- | --- | --- | --- |']
    repro = {}
    for m in R:
        old = [json.loads((E68 / f'{m}_s{s}' / 'metrics.json').read_text())['dev_stochastic'] for s in range(a.seeds)]
        for tag, rows in (('250 (e68)', old), ('500 (e69)', [r['dev_stochastic'] for r in R[m]])):
            L.append(f"| {LABEL[m]} | {tag} | {np.mean([x['air_m3'] for x in rows]):,.0f} | {np.mean([x['over_budget_steps'] for x in rows]):.3f} | "
                     f"{np.mean([x['day_violation_pct'] for x in rows]):.2f} | {np.mean([x['nh4_excess_mg_h_L'] for x in rows]):.2f} |")
        same = []
        for s in range(a.seeds):
            a_log = json.loads((E68 / f'{m}_s{s}' / 'train_log.json').read_text())
            b_log = logs[m][s][:len(a_log)]
            same.append(all(abs(x['cost'] - y['cost']) < 1e-9 and abs(x['air_m3'] - y['air_m3']) < 1e-6 for x, y in zip(a_log, b_log)))
        repro[m] = sum(same)
    J['reproduce_e68_first_250'] = repro
    L += ['', '재현 검사: 처음 250번 갱신의 학습 수집 (비용, 공기) 이 e68 과 같은 seed 수 ' + ', '.join(f'{LABEL[m]} {v}/{a.seeds}' for m, v in repro.items()) + '.',
          '', 'holdout 과 test 는 이 실험에서 쓰지 않았다.']
    (HERE / 'results.md').write_text('\n'.join(L) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(dict(judgement=J, root=str(root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=10)
    main(p.parse_args())
