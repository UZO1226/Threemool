"""실행 결과에서 표와 요약을 만든다. 학습이나 체크포인트 선택을 하지 않는다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Lee/code/bsm'))
import bsm_binary as cost_modes  # noqa: E402

MODES = ('mean', 'binary')
METHODS = ('ppo', 'fcsrl')
SEEDS = (0, 1, 2)
LABELS = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO'}
MODE_LABELS = {'mean': '평균 비용 (Yoo/e17 방식)', 'binary': '이진 비용 (Lee 변경)'}
KEYS = ('aeration_kwh_per_day', 'flow_weighted_nh4', 'step_violation_percent',
        'violating_steps_per_day', 'nh4_violation_percent', 'daily_nh4_cvar20',
        'daily_violating_steps_cvar20')


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'sd': float(values.std(ddof=0)),
            'values': [float(v) for v in values]}


def welch(a, b):
    """탐색적 Welch 검정. 다중 비교 보정을 하지 않는다."""
    from scipy import stats as st

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    result = st.ttest_ind(a, b, equal_var=False)
    return {'t': float(result.statistic), 'p': float(result.pvalue)}


def collect(runs):
    rows = {}
    for mode in MODES:
        for method in METHODS:
            for seed in SEEDS:
                path = runs / mode / method / f'seed_{seed}' / 'metrics.json'
                rows[(mode, method, seed)] = json.loads(path.read_text())
    return rows


def table(rows, weather, stage='final'):
    out = {}
    for mode in MODES:
        for method in METHODS:
            runs = [rows[(mode, method, s)] for s in SEEDS]
            week = [r[stage][weather]['week'] for r in runs]
            entry = {k: stats([w[k] for w in week]) for k in KEYS}
            entry['within_own_limit_seeds'] = sum(
                cost_modes.satisfied(mode, w) for w in week)
            if stage == 'final':
                initial = [r['initial'][weather]['week'] for r in runs]
                entry['delta_from_initial'] = {
                    k: stats([w[k] - i[k] for w, i in zip(week, initial)])
                    for k in ('aeration_kwh_per_day', 'step_violation_percent', 'flow_weighted_nh4')}
            out[f'{mode}/{method}'] = entry
    return out


def markdown(rows, weather, stage='final'):
    data = table(rows, weather, stage)
    lines = ['| 비용 모드 | 방법 | 폭기 kWh/day | 평균 NH4 | 순간 위반 % | 하루 위반 스텝 | 자기 한도 만족 seed |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for mode in MODES:
        for method in METHODS:
            d = data[f'{mode}/{method}']
            lines.append('| {} | {} | {:,.0f} ± {:,.0f} | {:.3f} ± {:.3f} | {:.2f} ± {:.2f} | {:.1f} ± {:.1f} | {}/3 |'.format(
                MODE_LABELS[mode], LABELS[method],
                d['aeration_kwh_per_day']['mean'], d['aeration_kwh_per_day']['sd'],
                d['flow_weighted_nh4']['mean'], d['flow_weighted_nh4']['sd'],
                d['step_violation_percent']['mean'], d['step_violation_percent']['sd'],
                d['violating_steps_per_day']['mean'], d['violating_steps_per_day']['sd'],
                d['within_own_limit_seeds']))
    return '\n'.join(lines), data


def comparisons(rows, weather):
    """각 비용 모드 안에서 FCSRL 과 PPO 의 차이, 그리고 모드 간 차이의 차이."""
    out = {}
    gaps = {}
    for mode in MODES:
        pairs = {}
        for key in ('step_violation_percent', 'aeration_kwh_per_day', 'flow_weighted_nh4',
                    'violating_steps_per_day'):
            fcsrl = [rows[(mode, 'fcsrl', s)]['final'][weather]['week'][key] for s in SEEDS]
            ppo = [rows[(mode, 'ppo', s)]['final'][weather]['week'][key] for s in SEEDS]
            pairs[key] = {'fcsrl_minus_ppo': float(np.mean(fcsrl) - np.mean(ppo)),
                          'welch': welch(fcsrl, ppo)}
            gaps.setdefault(key, {})[mode] = float(np.mean(fcsrl) - np.mean(ppo))
        out[mode] = pairs
    out['gap_difference_binary_minus_mean'] = {
        key: float(value['binary'] - value['mean']) for key, value in gaps.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = collect(a.runs)
    manifest = json.loads((a.runs / 'manifest.json').read_text())
    references = json.loads((a.prepared / 'references.json').read_text())

    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'code_commit': manifest['code_commit'], 'upstream_commit': manifest['upstream_commit'],
        'updates_per_run': manifest['updates_per_run'], 'steps_per_run': manifest['steps_per_run'],
        'seeds': list(SEEDS), 'limits': {m: cost_modes.daily_limit(m) for m in MODES},
        'prepared_metadata_sha256': hashlib.sha256((a.prepared / 'metadata.json').read_bytes()).hexdigest(),
        'references': {name: {w: {k: value[w]['week'][k] for k in KEYS} for w in ('dry', 'rain')}
                       for name, value in references.items()},
        'final': {w: table(rows, w) for w in ('dry', 'rain')},
        'initial': {w: table(rows, w, 'initial') for w in ('dry', 'rain')},
        'comparisons': {w: comparisons(rows, w) for w in ('dry', 'rain')},
        'training': {f'{mode}/{method}/seed_{seed}': {
            'lambda_last': rows[(mode, method, seed)]['training'][-1]['lambda'],
            'daily_cost_mean_first': rows[(mode, method, seed)]['training'][0]['daily_cost_mean'],
            'daily_cost_mean_last': rows[(mode, method, seed)]['training'][-1]['daily_cost_mean'],
            'train_step_violation_percent_last':
                rows[(mode, method, seed)]['training'][-1]['train_step_violation_percent'],
            'max_full_update_kl': max(r['full_update_probe_kl'] for r in rows[(mode, method, seed)]['training']),
            'max_gradient_norm': max(r['gradient_norm_max'] for r in rows[(mode, method, seed)]['training']),
            'max_feasibility_outside_fraction':
                max(r['feasibility_symlog_outside_fraction'] for r in rows[(mode, method, seed)]['training']),
            'elapsed_seconds': rows[(mode, method, seed)]['elapsed_seconds']}
            for mode in MODES for method in METHODS for seed in SEEDS}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    for weather in ('dry', 'rain'):
        text, _ = markdown(rows, weather)
        print(f'\n== {weather} (학습 후 정책, seed 3개 평균 ± 표준편차)')
        print(text)
    print('\n== FCSRL - PPO 차이')
    print(json.dumps(summary['comparisons'], indent=2, ensure_ascii=False))
    print('\nREPORTED', flush=True)


if __name__ == '__main__':
    main()
