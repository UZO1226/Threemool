"""실행 전 기준을 바꾸지 않고 seed 단위 요약과 같은 공기량 대조를 판정한다."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

METHODS = ('ppo', 'fcsrl', 'fcsrl_preserve')


def stats(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5 and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(t.ppf(.975, 4) * sd / np.sqrt(5))
    return {'mean': mean, 'std': sd, 'ci95': [mean - half, mean + half], 'values': x.tolist()}


def main(a):
    raw = {tag: {} for tag in ('mlp', 'gbm')}
    for method in METHODS:
        for tag in raw:
            rows = [json.loads((a.root / 'runs' / method / f'seed_{s}' / 'cross_evaluation.json').read_text())[tag]['summary'] for s in range(5)]
            raw[tag][method] = {k: stats([r[k] for r in rows]) for k in
                ('air_m3', 'cost', 'reward', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')}
            raw[tag][method]['feasible_seeds'] = sum(r['cost'] <= 15 for r in rows)
    matches = json.loads(a.matches.read_text())['matches']
    controls, gates = {}, {}
    lines = ['# e25 결과', '', '각 수치는 마지막 정책의 val 215에피소드 평균을 구한 뒤 seed 5개 사이 평균 ± 표본 표준편차로 집계했다.', '']
    for tag in raw:
        lines += [f'## {tag.upper()}', '', '| 방법 | 공기량 (Nm³/일) | 하루 비용 | 비용 15 이하 seed |', '| --- | --- | --- | --- |']
        for method in METHODS:
            r = raw[tag][method]
            lines.append(f"| {method} | {r['air_m3']['mean']:.0f} ± {r['air_m3']['std']:.0f} | {r['cost']['mean']:.2f} ± {r['cost']['std']:.2f} | {r['feasible_seeds']}/5 |")
        rs = sorted([r for r in matches if r['model'] == tag], key=lambda r: r['seed'])
        assert [r['seed'] for r in rs] == list(range(5))
        diff = stats([r['cost_difference'] for r in rs])
        matched = stats([r['matched']['cost'] for r in rs])
        ext = raw[tag]['fcsrl_preserve']
        reduction = 100 * (1 - ext['cost']['mean'] / matched['mean']) if matched['mean'] else 0.
        controls[tag] = {'paired_cost_difference': diff, 'matched_base_cost': matched,
                         'mean_cost_reduction_percent': reduction,
                         'max_abs_air_error_percent': max(abs(r['air_mismatch_percent']) for r in rs)}
        gates[tag] = {'extension_mean_cost_le_15': ext['cost']['mean'] <= 15,
                      'at_least_four_feasible_seeds': ext['feasible_seeds'] >= 4,
                      'all_air_matches_within_0_1_percent': all(r['matched_within_tolerance'] for r in rs),
                      'mean_cost_reduction_at_least_10_percent': reduction >= 10,
                      'paired_ci95_upper_negative': diff['ci95'][1] < 0}
        lines += ['', f"같은 공기량 기본 FCSRL 비용: {matched['mean']:.2f} ± {matched['std']:.2f}. 확장 비용 변화: {diff['mean']:.2f}, paired t 95% CI [{diff['ci95'][0]:.2f}, {diff['ci95'][1]:.2f}]. 평균 비용 감소율 {reduction:.2f}%.", '',
                  '| seed | 확장 비용 | 같은 공기량 기본 비용 | 차이 | 공기량 오차 (%) |', '| --- | --- | --- | --- | --- |']
        for r in rs:
            lines.append(f"| {r['seed']} | {r['extension']['cost']:.3f} | {r['matched']['cost']:.3f} | {r['cost_difference']:.3f} | {r['air_mismatch_percent']:.5f} |")
        lines += ['', '판정: ' + ', '.join(f'{k}={v}' for k, v in gates[tag].items()), '']
    passed = all(v for g in gates.values() for v in g.values())
    lines += ['## 전체 판정', '', f"사전 기준 전체 통과: {passed}.", '',
              '검증 데이터에서 공기량을 맞춘 사후 진단이다. 두 대리 모델은 실제 위반 과소예측 편향을 공유하며, 실제 공정 안전이나 절감을 입증하지 않는다. 이미 반복 사용한 val의 탐색적 결과이며 test를 평가하지 않았다.', '']
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'summary.json').write_text(json.dumps({'raw': raw, 'controls': controls, 'gates': gates, 'passed': passed}, indent=2, allow_nan=False))
    (a.out / 'results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'matches', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
