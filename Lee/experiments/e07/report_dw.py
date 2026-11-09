"""Lee/e07, e08 결과 집계 (DARROW 환경). 학습이나 모델 선택을 다시 하지 않고 저장된 metrics.json 만 읽는다.

원래 FCSRL(ppolag_fcsrl)을 기준으로 각 방법의 차이와 탐색적 Welch 검정을 낸다.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import stats

LABELS = {'ppolag_fcsrl': '원래 FCSRL + PPO-Lag', 'ppolag': 'PPO-Lag (참고)',
          'ppolag_fcsrl_pid3': 'FCSRL + 승수 반응 3배', 'ppolag_fcsrl_pid8': 'FCSRL + 승수 반응 8배'}
ORDER = list(LABELS)
KEYS = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')
LIMIT = 12.0


def load(runs):
    out = {}
    for label in ORDER:
        rows = []
        for f in sorted((runs / label).glob('seed_*/metrics.json')) if (runs / label).exists() else []:
            rows.append(json.loads(f.read_text()))
        if rows:
            out[label] = rows
    return out


def stat(values):
    v = np.asarray(values, float)
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=0)), 'values': [float(x) for x in v]}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2 or (a.std() == 0 and b.std() == 0):
        return None
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.pvalue)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = load(a.runs)
    manifest = json.loads((a.runs / 'manifest.json').read_text())
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'code_commit': manifest['code_commit'],
               'upstream_commit': manifest['upstream_commit'], 'steps_per_run': manifest['steps_per_run'],
               'data_sha256': manifest['data_sha256'], 'surrogate_sha256': manifest['surrogate_sha256'],
               'cost_limit': LIMIT, 'selected': {}, 'final': {}, 'feasible_selected_seeds': {}}
    for stage, key in (('selected', 'val_best'), ('final', 'val_final')):
        for label, rows in data.items():
            summary[stage][label] = {k: stat([r[key][k] for r in rows if k in r[key]]) for k in KEYS}
            summary[stage][label]['seeds'] = len(rows)
            summary[stage][label]['within_limit_seeds'] = int(sum(r[key]['cost'] <= LIMIT for r in rows))
    base = data.get('ppolag_fcsrl')
    summary['vs_original'] = {}
    if base:
        for stage, key in (('selected', 'val_best'), ('final', 'val_final')):
            summary['vs_original'][stage] = {}
            for label, rows in data.items():
                if label == 'ppolag_fcsrl':
                    continue
                d = {}
                for k in ('cost', 'air_m3', 'nh4_viol_pct'):
                    x, y = [r[key][k] for r in rows], [r[key][k] for r in base]
                    d[k] = {'diff': float(np.mean(x) - np.mean(y)), 'rel_percent': float(100 * (np.mean(x) / np.mean(y) - 1)),
                            'welch_p': welch(x, y),
                            'paired_diff_by_seed': [float(a_ - b_) for a_, b_ in zip(x, y)] if len(x) == len(y) else None}
                summary['vs_original'][stage][label] = d
    # 벌점이 기본 이점의 몇 배로 들어갔는지(명목 weight 와 같아야 한다)와 lambda 의 학습 중 추이
    summary['fpen_diagnostics'] = {}
    for label in data:
        if 'fpen' not in label:
            continue
        ratios, lam = [], []
        for f in sorted((a.runs / label).glob('seed_*/train_log.json')):
            log = json.loads(f.read_text())
            ratios += [x['fpen_to_base_rms'] for x in log if 'fpen_to_base_rms' in x]
            lam.append(log[-1]['lambda'])
        if ratios:
            summary['fpen_diagnostics'][label] = {
                'penalty_to_base_rms_median': float(np.median(ratios)),
                'penalty_to_base_rms_min': float(np.min(ratios)), 'penalty_to_base_rms_max': float(np.max(ratios)),
                'lambda_last_mean': float(np.mean(lam))}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    for stage, title in (('selected', '선택 모델 (검증 한도 12 만족 중 보상 최대)'), ('final', '마지막 모델')):
        print(f'\n== {title}, seed 평균 ± 표준편차')
        print('| 방법 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 에피소드 % | 한도 만족 seed |')
        print('| --- | --- | --- | --- | --- | --- | --- |')
        for label in data:
            s = summary[stage][label]
            print('| {} | {:,.0f} ± {:,.0f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} | {:.1f} | {}/{} |'.format(
                LABELS[label], s['air_m3']['mean'], s['air_m3']['sd'], s['cost']['mean'], s['cost']['sd'],
                s['nh4_viol_pct']['mean'], s['nh4_viol_pct']['sd'], s['do_viol_pct']['mean'],
                s['days_within_limit_pct']['mean'], s['within_limit_seeds'], s['seeds']))
    if summary['vs_original']:
        print('\n== 원래 FCSRL 대비 차이')
        for stage in ('selected', 'final'):
            for label, d in summary['vs_original'][stage].items():
                print(f'  [{stage}] {LABELS[label]}: 비용 {d["cost"]["diff"]:+.2f} ({d["cost"]["rel_percent"]:+.1f}%, p={d["cost"]["welch_p"]}), '
                      f'공기량 {d["air_m3"]["rel_percent"]:+.1f}% (p={d["air_m3"]["welch_p"]})')
    print('REPORTED', flush=True)


if __name__ == '__main__':
    main()
