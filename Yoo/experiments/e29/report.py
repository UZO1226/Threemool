"""고정한 다섯 조건을 seed 단위로 집계한다."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

METHODS = ('fcsrl', 'fcsrl_actor', 'fcsrl_critic', 'fcsrl_preserve', 'fcsrl_extra_ppo')
COMPARATORS = ('fcsrl_actor', 'fcsrl_critic', 'fcsrl_extra_ppo')


def stats(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5 and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(t.ppf(.975, 4) * sd / np.sqrt(5))
    p = float(2 * t.sf(abs(mean / (sd / np.sqrt(5))), 4)) if sd else (1. if mean == 0 else 0.)
    return {'mean': mean, 'std': sd, 'ci95': [mean - half, mean + half],
            'p_two_sided': p, 'values': x.tolist()}


def main(a):
    raw, mechanism = {tag: {} for tag in ('mlp', 'gbm')}, {}
    runs = {}
    for method in METHODS:
        runs[method] = []
        for seed in range(5):
            path = a.root / 'runs' / method / f'seed_{seed}'
            metrics = json.loads((path / 'metrics.json').read_text())
            assert metrics['steps'] == 202048 and metrics['method'] == method and metrics['seed'] == seed
            cross = json.loads((path / 'cross_evaluation.json').read_text())
            log = json.loads((path / 'train_log.json').read_text())
            runs[method].append({'seed': seed, 'metrics': metrics, 'cross': cross, 'log': log})
        for tag in raw:
            rows = [r['cross'][tag]['summary'] for r in runs[method]]
            raw[tag][method] = {k: stats([r[k] for r in rows]) for k in
                ('air_m3', 'cost', 'reward', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')}
            raw[tag][method]['feasible_seeds'] = sum(r['cost'] <= 15 for r in rows)
        entries = []
        for run in runs[method]:
            log = run['log']
            row = {'seed': run['seed'], 'elapsed_seconds': run['metrics']['elapsed_seconds'],
                   'accepted_updates': sum(r.get('preserve_accepted', 0) for r in log),
                   'pi_epochs_sum': sum(r['pi_epochs'] for r in log),
                   'before_kl_sum': sum(r.get('preserve_before_kl', r['encoder_sync_kl']) for r in log),
                   'after_kl_sum': sum(r.get('preserve_after_kl', r['encoder_sync_kl']) for r in log),
                   'before_value_error_sum': sum(r['preserve_before_value_mse'] for r in log) if 'preserve_before_value_mse' in log[0] else None,
                   'after_value_error_sum': sum(r['preserve_after_value_mse'] for r in log) if 'preserve_after_value_mse' in log[0] else None,
                   'extra_selected_steps_sum': sum(r.get('preserve_step', 0) for r in log),
                   'extra_wall_seconds': sum(r['extra_wall_seconds'] for r in log) if 'extra_wall_seconds' in log[0] else None,
                   'extra_cpu_seconds': sum(r['extra_cpu_seconds'] for r in log) if 'extra_cpu_seconds' in log[0] else None,
                   'extra_attempted_steps': sum(r['extra_attempted_steps'] for r in log) if 'extra_attempted_steps' in log[0] else None}
            entries.append(row)
        mechanism[method] = entries
    matches = json.loads(a.matches.read_text())['matches']
    assert len(matches) == 40
    comparisons, primary = {}, []
    for tag in raw:
        comparisons[tag] = {}
        for method in ('fcsrl', *COMPARATORS):
            rows = sorted([r for r in matches if r['model'] == tag and r['comparator'] == method], key=lambda r: r['seed'])
            assert [r['seed'] for r in rows] == list(range(5))
            result = stats([r['cost_difference'] for r in rows])
            result['all_air_matches_within_0_1_percent'] = all(r['matched_within_tolerance'] for r in rows)
            result['max_abs_air_mismatch_percent'] = max(abs(r['air_mismatch_percent']) for r in rows)
            result['matched_cost'] = stats([r['matched']['cost'] for r in rows])
            result['full_better_seed_count'] = sum(r['cost_difference'] < 0 for r in rows)
            comparisons[tag][method] = result
            if method in COMPARATORS:
                primary.append(result)
    running = 0.
    for rank, result in enumerate(sorted(primary, key=lambda x: x['p_two_sided'])):
        running = max(running, min(1., (len(primary) - rank) * result['p_two_sided']))
        result['p_holm_six'] = running
        result['full_advantage_supported'] = (result['all_air_matches_within_0_1_percent']
                                               and result['mean'] < 0 and running < .05)
    md = ['# e29 출력 보정 기여 분리 결과', '',
          '고정 학습량 202,048전이 뒤 마지막 정책. 각 모델 val43일 x 구성원5개를 먼저 평균하고 정책 seed5개의 평균 ± 표본 SD를 보고한다. 기본/full 10개는 재사용, 세 조건 15개는 신규다. test를 새로 평가하지 않았다.', '']
    for tag in raw:
        md += [f'## {tag.upper()} 원시 정책 성능', '',
               '| 방법 | 공기량 Nm³/일 | 하루 비용 | 한도 만족 seed |', '| --- | --- | --- | --- |']
        for method in METHODS:
            r = raw[tag][method]
            md.append(f"| {method} | {r['air_m3']['mean']:.0f} ± {r['air_m3']['std']:.0f} | {r['cost']['mean']:.2f} ± {r['cost']['std']:.2f} | {r['feasible_seeds']}/5 |")
        md += ['', '### full 보정 공기량에 맞춘 대조', '',
               '음의 차이는 full 보정 비용이 더 낮다는 뜻이다. 95% CI는 대비별 비보정 구간이고 Holm p는 사전 세 대비 x 두 모델의 6개 검정이다. 기본 FCSRL 대비는 e25 재현 참고로 다중 검정군에서 제외한다.', '',
               '| 대조 정책 | full 빼기 대조 비용 | paired 95% CI | 양측 p | Holm p | full 비용이 낮은 seed |',
               '| --- | --- | --- | --- | --- | --- |']
        for method, r in comparisons[tag].items():
            adjusted = f"{r['p_holm_six']:.4f}" if 'p_holm_six' in r else '참고'
            md.append(f"| {method} | {r['mean']:+.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['p_two_sided']:.4f} | {adjusted} | {r['full_better_seed_count']}/5 |")
        md += ['', '### seed별 원시 공기량과 비용', '', '| 방법 | seed | 공기량 | 비용 |', '| --- | --- | --- | --- |']
        for method in METHODS:
            for run in runs[method]:
                r = run['cross'][tag]['summary']
                md.append(f"| {method} | {run['seed']} | {r['air_m3']:.1f} | {r['cost']:.3f} |")
    md += ['', '## 보정 작동과 계산량', '',
           'KL과 가치 오차는 각 실행의 22갱신 합계를 seed 평균했다. 기본군 KL은 probe 표본이 달라 직접 크기 비교하지 않는다. extra-PPO의 teacher 오차 증가는 출력 보존이 목적이 아니므로 그 자체가 오류는 아니다. 실행 시간은 평가 포함이며 과거 재사용 결과와 서버 부하가 다르다.', '',
           '| 방법 | 수용 갱신/22 | KL 보정 전 합 | KL 보정 후 합 | 가치 오차 전 합 | 가치 오차 후 합 | 전체 시간 초 | 추가 CPU 초 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for method, rows in mechanism.items():
        avg = lambda key: float(np.mean([r[key] for r in rows]))
        cpu = f"{avg('extra_cpu_seconds'):.2f}" if rows[0]['extra_cpu_seconds'] is not None else '기록 없음'
        value_before = f"{avg('before_value_error_sum'):.4f}" if rows[0]['before_value_error_sum'] is not None else '측정 안 함'
        value_after = f"{avg('after_value_error_sum'):.4f}" if rows[0]['after_value_error_sum'] is not None else '측정 안 함'
        md.append(f"| {method} | {avg('accepted_updates'):.1f} | {avg('before_kl_sum'):.5f} | {avg('after_kl_sum'):.5f} | {value_before} | {value_after} | {avg('elapsed_seconds'):.1f} | {cpu} |")
    md += ['', '## 사전 대비 판정', '']
    for method in COMPARATORS:
        flags = {tag: comparisons[tag][method]['full_advantage_supported'] for tag in raw}
        md.append(f'- full 대 {method}: 같은 공기량의 음의 비용 차이와 Holm p<0.05, 두 모델 모두 충족 = {all(flags.values())}. 모델별 {flags}.')
    md += ['', '차이가 유의하지 않다는 것은 동등성의 증명이 아니다. 현재 비교는 구성 요소 및 추가 학습 패키지의 기여를 좁히는 탐색이다. 기존 e25 채택 기준과 원 환경의 실측 위반 과소예측 한계는 그대로이며, 실제 공정 안전이나 절감을 입증하지 않는다.', '']
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'summary.json').write_text(json.dumps({'raw': raw, 'comparisons': comparisons, 'mechanism': mechanism}, indent=2, allow_nan=False))
    (a.out / 'results.md').write_text('\n'.join(md))
    print('\n'.join(md[:30]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'matches', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
