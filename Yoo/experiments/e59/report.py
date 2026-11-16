"""e59 집계: 조건별 표, 사전 판정 J1~J4, 다음 단계 규칙."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEYS = ('air_m3', 'nh4_cost', 'do_cost', 'cost', 'nh4_excess_mg_h_L', 'worst_day_nh4_cost', 'days_within_limit_pct',
        'intervention_pct', 'lowered_pct', 'conflict_pct', 'policy_air_m3')
LABEL = {'ppo': 'PPO-Lag (e50)', 'fcsrl': 'FCSRL (e50)', 'fcsrl_actor': 'FCSRL + Actor 보정 (e50)', 'severity': 'FCSRL + severity (e58)',
         'record': '기록 운전', 'zero': '공기 0 정책 (차단만)'}


def load(root):
    out = {}
    for p in sorted(root.glob('runs/*/*/result.json')):
        r = json.loads(p.read_text())
        out[(r['policy'], r['seed'], r['shield'])] = r['val']
    return out


def ms(xs, nd=2):
    xs = np.asarray(xs, float)
    return f'{xs.mean():,.{nd}f}' if len(xs) == 1 else f'{xs.mean():,.{nd}f} ± {xs.std():,.{nd}f}'


def row(label, vals):
    g = lambda k, nd=2: ms([v.get(k, 0.0) for v in vals], nd)  # noqa: E731
    return (f"| {label} | {g('air_m3', 0)} | {g('nh4_cost')} | {g('do_cost')} | {g('cost')} | {g('nh4_excess_mg_h_L')} | "
            f"{g('worst_day_nh4_cost', 1)} | {g('days_within_limit_pct', 1)} | {g('intervention_pct', 1)} | {g('lowered_pct', 1)} |")


def main(a):
    R = load(a.root)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    policies, seeds, shields = protocol['policies'], protocol['seeds'], protocol['shields']
    L = ['# e59 결과: Park 모델 예측 차단을 Yoo 정책에 적용', '',
         'e50 공통 v1.1 환경, 검증 40일 x 구성원 5개, 결정적 행동, 마지막 정책. 새 학습 없음. 현장 또는 독립 test 결과가 아니다.',
         '차단은 Park/code/min_brake.py 를 수정 없이 복사해 실행 시점에만 걸었다. 정책은 차단 없이 학습된 것이다.', '']
    head = ['| 조건 | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | NH4 초과 mg h/L | 최악 날짜 NH4 | 한도 12 날짜 % | 개입 % | 내린 % |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for sh in shields:
        L += [f'## 차단: {sh}', ''] + head
        for pol in policies:
            L.append(row(LABEL[pol], [R[(pol, s, sh)] for s in seeds]))
        for pol in protocol['fixed']:
            L.append(row(LABEL[pol], [R[(pol, 0, sh)]]))
        L.append('')
    summary = {'means': {}, 'seed_level': {}}
    for sh in shields:
        for pol in policies + protocol['fixed']:
            vals = [R[(pol, s, sh)] for s in (seeds if pol in policies else [0])]
            summary['means'][f'{pol}|{sh}'] = {k: float(np.mean([v.get(k, 0.0) for v in vals])) for k in KEYS}
            summary['seed_level'][f'{pol}|{sh}'] = [{k: v.get(k, 0.0) for k in KEYS} for v in vals]
    # J1: 정책 기여
    zero = R[('zero', 0, 'park_e19')]
    L += ['## J1 정책 기여 (정책 + park_e19 대 공기 0 정책 + park_e19)', '',
          f"공기 0 정책 + park_e19: 공기 {zero['air_m3']:,.0f}, 비용 {zero['cost']:.2f}, NH4 {zero['nh4_viol_pct']:.2f}%, DO {zero['do_viol_pct']:.2f}%", '',
          '| 방법 | 공기와 비용 모두 낮은 seed | 공기만 낮은 seed | 비용만 낮은 seed | 판정 |', '| --- | --- | --- | --- | --- |']
    j1 = {}
    for pol in policies:
        both = air = cost = 0
        for s in seeds:
            v = R[(pol, s, 'park_e19')]
            a_, c_ = v['air_m3'] < zero['air_m3'], v['cost'] < zero['cost']
            both += a_ and c_
            air += a_
            cost += c_
        verdict = '기여' if both >= 4 else ('기여 없음' if both == 0 else '불확정')
        j1[pol] = dict(both=both, air_only=air, cost_only=cost, verdict=verdict)
        L.append(f'| {LABEL[pol]} | {both}/5 | {air}/5 | {cost}/5 | {verdict} |')
    # J2: 차단 효과 (같은 seed)
    L += ['', '## J2 같은 seed 에서 차단의 효과 (park_e19 minus none)', '', '| 방법 | 공기량 변화 % | NH4 비용 변화 | DO 비용 변화 | 합산 비용 변화 |', '| --- | --- | --- | --- | --- |']
    j2 = {}
    for pol in policies:
        d = {k: [R[(pol, s, 'park_e19')][k] - R[(pol, s, 'none')][k] for s in seeds] for k in ('air_m3', 'nh4_cost', 'do_cost', 'cost')}
        base_air = np.mean([R[(pol, s, 'none')]['air_m3'] for s in seeds])
        j2[pol] = {k: float(np.mean(v)) for k, v in d.items()}
        j2[pol]['air_pct'] = 100 * j2[pol]['air_m3'] / base_air
        L.append(f"| {LABEL[pol]} | {j2[pol]['air_pct']:+.2f} | {ms(d['nh4_cost'])} | {ms(d['do_cost'])} | {ms(d['cost'])} |")
    # J3: 교환선
    x = np.array([R[(pol, s, 'none')]['air_m3'] for pol in policies for s in seeds]) / 1000
    y = np.array([R[(pol, s, 'none')]['nh4_cost'] for pol in policies for s in seeds])
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    L += ['', '## J3 교환선 (none 조건 20개 정책에 맞춘 직선) 대비 잔차', '',
          f'NH4 비용 = {intercept:.2f} + {slope:.3f} x 공기(천 m³), r = {r:.2f}', '', '| 조건 | 공기량 | NH4 비용 | 직선 예측 | 잔차 |', '| --- | --- | --- | --- | --- |']
    j3 = {}
    for sh in shields[1:]:
        for pol in policies + protocol['fixed']:
            m = summary['means'][f'{pol}|{sh}']
            pred = intercept + slope * m['air_m3'] / 1000
            j3[f'{pol}|{sh}'] = float(m['nh4_cost'] - pred)
            L.append(f"| {LABEL[pol]} + {sh} | {m['air_m3']:,.0f} | {m['nh4_cost']:.2f} | {pred:.2f} | {m['nh4_cost'] - pred:+.2f} |")
    rec = summary['means']['record|none']
    L.append(f"| 기록 운전 | {rec['air_m3']:,.0f} | {rec['nh4_cost']:.2f} | {intercept + slope * rec['air_m3'] / 1000:.2f} | {rec['nh4_cost'] - (intercept + slope * rec['air_m3'] / 1000):+.2f} |")
    # J4: Park 대비
    park = protocol['park_reference_same_environment']
    best = min(policies, key=lambda p: summary['means'][f'{p}|park_e19']['cost'])
    bm = summary['means'][f'{best}|park_e19']
    L += ['', '## J4 Park 같은 환경 결과와 나란히 (학습 조건이 달라 우열 아님)', '', '| 조건 | 공기량 | 합산 비용 |', '| --- | --- | --- |',
          f"| 공기 0 정책 + park_e19 (Yoo e59) | {zero['air_m3']:,.0f} | {zero['cost']:.2f} |",
          f"| {LABEL[best]} + park_e19 (Yoo e59, 비용 최저 방법 평균) | {bm['air_m3']:,.0f} | {bm['cost']:.2f} |"]
    for k, v in park.items():
        L.append(f"| Park {k} | {v['air_m3']:,} | {v['cost']:.2f} |")
    e22 = park['e22_H1_4x16_ensemble_plus_e19_shield']
    model_only_beats = bool(zero['air_m3'] <= e22['air_m3'] and zero['cost'] <= e22['cost'])
    nxt = '모델 예측 제어기 개선' if model_only_beats else 'Park e12 방식으로 Yoo FCSRL/PPO 를 차단과 함께 학습 (e60)'
    L += ['', '## 다음 단계 (사전 규칙)', '', f'공기 0 정책 + park_e19 가 Park e22 H1 을 공기와 비용 모두에서 넘는가: {model_only_beats}. 따라서 다음은 {nxt}.', '',
          '반복 사용한 개발 val 의 탐색 결과이며 통계적 확증이나 공식 채택을 주장하지 않는다.']
    summary.update(J1=j1, J2=j2, J3=dict(slope=float(slope), intercept=float(intercept), r=r, residuals=j3),
                   J4=dict(zero_plus_e19={k: zero[k] for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct')}, best_policy=best, park=park),
                   model_only_beats_park_e22=model_only_beats, next_step=nxt, root=str(a.root))
    (a.out / 'results.md').write_text('\n'.join(L) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE)
    main(p.parse_args())
