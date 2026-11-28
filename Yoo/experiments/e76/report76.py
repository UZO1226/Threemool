"""e76 모든 seed와 맞춤 실패를 보존하며 선택 궤적을 독립 검산한다."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

KEYS = ('air_m3', 'step_cost', 'nh4_excess_mg_h_L', 'do_step_viol_pct', 'cost_limit_violation_pct')
SAFETY = KEYS[1:4]


def audit(job):
    path = Path(job)
    row = json.loads((path / 'result.json').read_text())
    assert hashlib.sha256((path / 'selection.json').read_bytes()).hexdigest() == row['selection_sha']
    for split in ('dev', 'val'):
        for version in ('native', 'selected'):
            with np.load(path / f'{split}_{version}.npz') as z:
                state, air, zones = z['states'], z['air'], z['zones']
                nh4 = state[..., 0]
                cost = ((nh4 > 4) | (np.cumsum(nh4, -1) > 380)).sum(-1)
                direct = dict(air_m3=air.sum(-1) / 4, step_cost=cost,
                              nh4_excess_mg_h_L=np.maximum(nh4 - 4, 0).sum(-1) / 4,
                              do_step_viol_pct=100 * (state[..., 2:].max(-1) > 2).mean(-1),
                              cost_limit_violation_pct=100 * (cost > 4.75))
                assert np.allclose(air, zones.sum(-1), atol=1e-9, rtol=0)
                for key, values in direct.items():
                    assert np.allclose(values, z[key], atol=1e-9, rtol=0), key
                    assert np.isclose(values.mean(), row[version][split]['mean'][key], atol=1e-9, rtol=0), key
    return row


def main(root):
    rows = [audit(root / 'jobs' / f'{m}_s{s}') for s in range(10) for m in ('fcsrl', 'ppolag')]
    out = dict(rows=rows, pairs={}, summary={}, all_80_rollouts_audited=True,
               independent_confirmatory_test=False, holdout_evaluated=False, test_evaluated=False)
    md = ['# e76 같은 공기 사용량에서의 표현 학습 진단', '',
          'e74 저장 정책의 배율 조정 후 평가다. dev/val은 반복 사용한 개발 자료이며 새 학습과 독립 확인은 아니다. '
          '공기 m³/23.75h, 비용 스텝/일, NH4 초과 mg h/L, DO 위반 %. 표는 정책 seed 10개 평균이다.', '',
          '| 분할 | 방법 | 배율 | 공기 | 비용 | NH4 초과 | DO 위반 % | 비용 한도 충족 seed | 에피소드 비용 한도 초과 % |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    baselines = json.loads(Path('/home/jio/fcsrl_src/results/e74/baselines/baselines.json').read_text())['chosen']['abac']
    for split in ('dev', 'val'):
        for method in ('fcsrl', 'ppolag'):
            group = [r for r in rows if r['method'] == method]
            for version in ('native', 'selected'):
                means = {k: float(np.mean([r[version][split]['mean'][k] for r in group])) for k in KEYS}
                std = {k: float(np.std([r[version][split]['mean'][k] for r in group], ddof=1)) for k in KEYS}
                count = sum(r[version][split]['mean']['step_cost'] <= 4.75 for r in group)
                out['summary'][f'{split}/{method}/{version}'] = dict(mean=means, std=std, limit_seeds=count)
                md.append(f"| {split} | {method} | {version} | {means['air_m3']:.0f} | {means['step_cost']:.3f} | "
                          f"{means['nh4_excess_mg_h_L']:.3f} | {means['do_step_viol_pct']:.3f} | {count}/10 | {means['cost_limit_violation_pct']:.2f} |")
        b = baselines[split]
        md.append(f"| {split} | ABAC 참고 | 원 설정 | {b['air_m3']:.0f} | {b['step_cost']:.3f} | {b['nh4_excess_mg_h_L']:.3f} | {b['do_step_viol_pct']:.3f} | 해당 없음 | 미집계 |")
    md += ['', '## seed별 맞춤과 짝 비교', '',
           '| 분할 | seed | F 배율 | P 배율 | F 목표오차 % (dev) | P 목표오차 % (dev) | 짝 공기차 % | 비교 가능 | 비용 차 F-P | NH4 초과 차 | DO 차 %p | 공동 개선 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for split in ('dev', 'val'):
        pairs = []
        for seed in range(10):
            f = next(r for r in rows if r['method'] == 'fcsrl' and r['seed'] == seed)
            p = next(r for r in rows if r['method'] == 'ppolag' and r['seed'] == seed)
            fm, pm = f['selected'][split]['mean'], p['selected'][split]['mean']
            air_pct = 200 * (fm['air_m3'] - pm['air_m3']) / (fm['air_m3'] + pm['air_m3'])
            eligible = bool(f['dev_target_matched'] and p['dev_target_matched'] and abs(air_pct) <= (1 if split == 'dev' else 3))
            # A failed dev pair cannot be rescued by val matching.
            if split == 'val':
                eligible = eligible and out['pairs']['dev'][seed]['eligible']
            diff = {k: fm[k] - pm[k] for k in KEYS}
            fwin = eligible and all(diff[k] <= 0 for k in SAFETY) and diff['step_cost'] < 0
            pwin = eligible and all(diff[k] >= 0 for k in SAFETY) and diff['step_cost'] > 0
            pair = dict(seed=seed, eligible=eligible, air_difference_pct=air_pct, difference=diff, fcsrl_win=fwin, ppolag_win=pwin)
            pairs.append(pair)
            label = 'FCSRL' if fwin else '고정 표현 PPO' if pwin else '없음'
            md.append(f"| {split} | {seed} | {f['alpha']} | {p['alpha']} | {f['target_error_pct']:+.2f} | {p['target_error_pct']:+.2f} | "
                      f"{air_pct:+.2f} | {eligible} | {diff['step_cost']:+.3f} | {diff['nh4_excess_mg_h_L']:+.3f} | {diff['do_step_viol_pct']:+.3f} | {label} |")
        out['pairs'][split] = pairs
        mean_diff = {k: float(np.mean([p['difference'][k] for p in pairs])) for k in KEYS}
        fw, pw = sum(p['fcsrl_win'] for p in pairs), sum(p['ppolag_win'] for p in pairs)
        fs = fw >= 8 and all(mean_diff[k] <= 0 for k in SAFETY) and mean_diff['step_cost'] < 0
        ps = pw >= 8 and all(mean_diff[k] >= 0 for k in SAFETY) and mean_diff['step_cost'] > 0
        out['summary'][f'{split}/verdict'] = dict(eligible=sum(p['eligible'] for p in pairs), fcsrl_wins=fw, ppolag_wins=pw,
                                                  fcsrl_signal=fs, ppolag_signal=ps, mean_difference=mean_diff)
    md += ['', '## 판정과 해석', '']
    for split in ('dev', 'val'):
        v = out['summary'][f'{split}/verdict']
        signal = 'FCSRL 개선 신호' if v['fcsrl_signal'] else '고정 표현 PPO 개선 신호' if v['ppolag_signal'] else '공동 개선 신호 미확인'
        md.append(f"- {split}: 비교 가능 {v['eligible']}/10, 공동 개선 FCSRL {v['fcsrl_wins']}/10, 고정 표현 PPO {v['ppolag_wins']}/10. {signal}.")
    md += ['', '비교 불가능한 짝도 표와 전체 평균에 남겼다. 맞춤 실패는 방법의 열등성 증거가 아니다. '
           '배율 조정은 정책을 바꾸므로 원 정책 자체의 성능과 구분한다. 전체 평균이 좋아도 모든 날짜의 안전을 뜻하지 않는다. '
           '모델 구성원과 표본 난수는 독립된 정책 seed가 아니다.', '',
           '검산: 원 e74 표본3개 집계 및 첫 표본 궤적 재현, 선택 배율 dev 재생, 모든 native/selected 궤적의 공기와 비용 직접 재계산 통과. '
           '80개 궤적 파일 각각 표본3개를 포함한다. 환경 타당성이나 실제 안전의 검증은 아니다.']
    (root / 'results.md').write_text('\n'.join(md) + '\n')
    (root / 'summary.json').write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({k: v for k, v in out['summary'].items() if k.endswith('verdict')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
