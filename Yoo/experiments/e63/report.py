"""e63 집계: J1 ~ J5."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABEL = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL'}
COND = {'total': '총 공기만', 'alloc': '총 공기 + 배분'}


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def train_p99(prepared):
    d = pd.read_pickle(Path(prepared) / 'data.pkl')
    zones = [f'air{i}' for i in range(1, 6)]
    rows = np.concatenate([d['data'].loc[day][zones].to_numpy(float) for day in d['days']['train']])
    return np.percentile(rows, 99, axis=0)


def decompose(states):
    v = states[:, :, 0] > 4
    carry = np.where(v.all(1), v.shape[1], np.argmin(v, axis=1))
    return float(carry.mean()), float((v.sum(1) - carry).mean())


def verdict(rows):
    both = sum(r['air_m3'] < 0 and r['cost'] < 0 for r in rows)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    if both >= 4 and mean['air_m3'] < 0 and mean['cost'] < 0:
        v = '개선'
    elif mean['cost'] < 0 and mean['air_m3'] > 0:
        v = '교환 (공기 증가)'
    elif mean['air_m3'] < 0 and mean['cost'] > 0:
        v = '교환 (비용 증가)'
    else:
        v = '개선 없음'
    return both, mean, v


def main(a):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    manifest = json.loads((a.root / 'manifest.json').read_text())
    p99 = train_p99(manifest['prepared'])
    methods, seeds = protocol['methods'], protocol['seeds']
    M, X = {}, {}
    for cond in ('total', 'alloc'):
        for m in methods:
            for s in seeds:
                run = a.root / 'runs' / cond / m / f'seed_{s}'
                M[(cond, m, s)] = json.loads((run / 'metrics.json').read_text())
                X[(cond, m, s)] = json.loads((a.root / 'cross' / cond / m / f'seed_{s}' / 'result.json').read_text())['val']
                z = np.load(run / 'val_park_e19_rollout.npz')
                if 'zones' in z.files:
                    zones = z['zones']
                else:  # 총 공기 조건은 기록 배분이므로 공기 x 기록 배분으로 복원한다
                    zones = None
                carry, ctrl = decompose(z['states'])
                M[(cond, m, s)]['val_park_e19'].update(nh4_carryover=carry, nh4_controllable=ctrl,
                                                       coverage_p99=None if zones is None else float((zones <= p99[None, None, :] + 1e-9).all(2).mean()))
    L = ['# e63 결과: 총 공기와 구간 배분을 함께 정하는 정책', '',
         'DO 대응 수정 모델 (e51) 에서 학습. 학습 중 차단 park_e10, 401,280전이, 마지막 정책, seed 5. 검증 40일 x 구성원 5개. 현장 또는 test 결과가 아니다.', '',
         '## 수정 모델, val park_e19', '',
         '| 방법 | 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | NH4 시작 이월 | NH4 제어 구간 | w 평균 | w>0 % | coverage p99 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for m in methods:
        for cond in ('total', 'alloc'):
            v = [M[(cond, m, s)]['val_park_e19'] for s in seeds]
            cov = ms([x['coverage_p99'] for x in v], 3) if cond == 'alloc' else '기록 배분'
            L.append(f"| {LABEL[m]} | {COND[cond]} | {ms([x['air_m3'] for x in v], 0)} | {ms([x['nh4_cost'] for x in v])} | {ms([x['do_cost'] for x in v])} | "
                     f"{ms([x['cost'] for x in v])} | {ms([x['nh4_carryover'] for x in v])} | {ms([x['nh4_controllable'] for x in v])} | "
                     f"{ms([x.get('w_mean', 0) for x in v], 3)} | {ms([x.get('w_positive_pct', 0) for x in v], 1)} | {cov} |")
    L += ['', '## 공통 v1.1 모델 교차 평가, val park_e19', '', '| 방법 | 조건 | 공기량 | NH4 비용 | DO 비용 | 합산 비용 | w 평균 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for m in methods:
        for cond in ('total', 'alloc'):
            v = [X[(cond, m, s)] for s in seeds]
            L.append(f"| {LABEL[m]} | {COND[cond]} | {ms([x['air_m3'] for x in v], 0)} | {ms([x['nh4_cost'] for x in v])} | {ms([x['do_cost'] for x in v])} | "
                     f"{ms([x['cost'] for x in v])} | {ms([x.get('w_mean', 0) for x in v], 3)} |")
    J = {}
    for tag, src, title in (('J1', lambda c, m, s: M[(c, m, s)]['val_park_e19'], 'J1 수정 모델'), ('J2', lambda c, m, s: X[(c, m, s)], 'J2 공통 모델 교차')):
        L += ['', f'## {title}: alloc minus total (seed 번호끼리)', '', '| 방법 | seed | 공기 변화 | 비용 변화 | NH4 변화 | DO 변화 |', '| --- | --- | --- | --- | --- | --- |']
        for m in methods:
            rows = []
            for s in seeds:
                n_, o_ = src('alloc', m, s), src('total', m, s)
                rows.append({k: n_[k] - o_[k] for k in ('air_m3', 'cost', 'nh4_cost', 'do_cost')})
                r = rows[-1]
                L.append(f"| {LABEL[m]} | {s} | {r['air_m3']:+,.0f} | {r['cost']:+.3f} | {r['nh4_cost']:+.3f} | {r['do_cost']:+.3f} |")
            both, mean, v = verdict(rows)
            J[f'{tag}_{m}'] = dict(paired=rows, mean=mean, both_lower_seeds=both, verdict=v)
            L.append(f"| {LABEL[m]} | 평균 | {mean['air_m3']:+,.0f} | {mean['cost']:+.3f} | {mean['nh4_cost']:+.3f} | {mean['do_cost']:+.3f} |")
            L.append(f"| {LABEL[m]} | 판정 | 둘 다 낮은 seed {both}/5 | {v} | | |")
    L += ['', '## 종합', '']
    for m in methods:
        j1, j2 = J[f'J1_{m}']['verdict'], J[f'J2_{m}']['verdict']
        cov = float(np.mean([M[('alloc', m, s)]['val_park_e19']['coverage_p99'] for s in seeds]))
        note = '이득이 DO 수정 모델에 기댐' if j1 == '개선' and j2 != '개선' else ''
        J[f'J3_{m}'] = dict(coverage_p99_mean=cov, outside_support=cov < 0.95)
        L.append(f"- {LABEL[m]}: J1 {j1}, J2 {j2}{', ' + note if note else ''}. coverage p99 평균 {cov:.3f}{' (지원 범위 밖 의존)' if cov < 0.95 else ''}.")
    L += ['', 'e60, e61 (공통 모델, 1차원) 과 Park 결과는 모델과 행동 공간이 달라 우열로 쓰지 않는다. 반복 사용한 개발 val 의 seed 5 탐색이다.']
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(dict(judgement=J, root=str(a.root)), indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
