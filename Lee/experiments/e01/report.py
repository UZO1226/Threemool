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

MODES = ('mean', 'binary', 'binary_do')
METHODS = ('ppo', 'fcsrl', 'fcsrl_fpen')
SEEDS = (0, 1, 2)
LABELS = {'ppo': 'PPO-Lag', 'fcsrl': 'FCSRL + PPO',
          'fcsrl_fpen': 'FCSRL + PPO + feasibility 벌점'}
MODE_LABELS = {'mean': '평균 비용 (Yoo/e17 방식)', 'binary': '이진 비용 (NH4)',
               'binary_do': '이진 비용 (NH4 와 DO)'}
KEYS = ('aeration_kwh_per_day', 'flow_weighted_nh4', 'step_violation_percent',
        'violating_steps_per_day', 'nh4_violation_percent', 'daily_nh4_cvar20',
        'daily_violating_steps_cvar20', 'do5_mean', 'do_step_violation_percent',
        'joint_step_violation_percent', 'joint_violating_steps_per_day')
COMPARE_KEYS = ('step_violation_percent', 'aeration_kwh_per_day', 'flow_weighted_nh4',
                'violating_steps_per_day', 'joint_step_violation_percent')


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
    """실제로 있는 조합만 읽는다. 실험마다 모드와 방법의 조합이 다르다."""
    rows = {}
    for mode in MODES:
        for method in METHODS:
            for seed in SEEDS:
                path = runs / mode / method / f'seed_{seed}' / 'metrics.json'
                if path.exists():
                    rows[(mode, method, seed)] = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f'No metrics.json found under {runs}')
    return rows


def present(rows):
    pairs = {(mode, method) for mode, method, _ in rows}
    return sorted(pairs, key=lambda x: (MODES.index(x[0]), METHODS.index(x[1])))


def table(rows, weather, stage='final'):
    out = {}
    for mode, method in present(rows):
            runs = [rows[(mode, method, s)] for s in SEEDS if (mode, method, s) in rows]
            week = [r[stage][weather]['week'] for r in runs]
            entry = {k: stats([w[k] for w in week])
                     for k in KEYS if all(k in w for w in week)}
            limit = runs[0].get('daily_cost_limit')
            entry['daily_cost_limit'] = limit
            entry['seeds'] = len(week)
            entry['within_own_limit_seeds'] = sum(
                cost_modes.satisfied(mode, w, limit) for w in week)
            if stage == 'final':
                initial = [r['initial'][weather]['week'] for r in runs]
                entry['delta_from_initial'] = {
                    k: stats([w[k] - i[k] for w, i in zip(week, initial)])
                    for k in ('aeration_kwh_per_day', 'step_violation_percent', 'flow_weighted_nh4')}
            out[f'{mode}/{method}'] = entry
    return out


def markdown(rows, weather, stage='final'):
    data = table(rows, weather, stage)
    lines = ['| 비용 모드 | 방법 | 폭기 kWh/day | 평균 NH4 | NH4 위반 % | DO 위반 % | 결합 위반 % | 한도 만족 seed |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    nan = float('nan')
    for mode, method in present(rows):
        d = data[f'{mode}/{method}']
        lines.append('| {} | {} | {:,.0f} ± {:,.0f} | {:.3f} ± {:.3f} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} | {}/{} |'.format(
            MODE_LABELS[mode], LABELS[method],
            d['aeration_kwh_per_day']['mean'], d['aeration_kwh_per_day']['sd'],
            d['flow_weighted_nh4']['mean'], d['flow_weighted_nh4']['sd'],
            d['step_violation_percent']['mean'], d['step_violation_percent']['sd'],
            d.get('do_step_violation_percent', {}).get('mean', nan),
            d.get('joint_step_violation_percent', {}).get('mean', nan),
            d['within_own_limit_seeds'], d['seeds']))
    return '\n'.join(lines), data


def comparisons(rows, weather):
    """각 비용 모드 안에서 FCSRL 과 PPO 의 차이, 그리고 모드 간 차이의 차이."""
    out = {}
    gaps = {}

    def series(mode, method, key):
        return [rows[(mode, method, s)]['final'][weather]['week'][key] for s in SEEDS
                if (mode, method, s) in rows
                and key in rows[(mode, method, s)]['final'][weather]['week']]

    modes = {mode for mode, _ in present(rows)}
    for mode in sorted(modes, key=MODES.index):
        methods = {method for m, method in present(rows) if m == mode}
        pairs = {}
        for other in sorted(methods - {'ppo'}):
            for key in COMPARE_KEYS:
                a, b = series(mode, other, key), series(mode, 'ppo', key)
                if not a or not b:
                    continue
                label = key if other == 'fcsrl' else f'{other}/{key}'
                pairs[label] = {'fcsrl_minus_ppo': float(np.mean(a) - np.mean(b)),
                                'welch': welch(a, b)}
                if other == 'fcsrl':
                    gaps.setdefault(key, {})[mode] = float(np.mean(a) - np.mean(b))
        if pairs:
            out[mode] = pairs
    if {'binary', 'mean'} <= modes:
        out['gap_difference_binary_minus_mean'] = {
            key: float(value['binary'] - value['mean']) for key, value in gaps.items()
            if {'binary', 'mean'} <= set(value)}
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
        'references': {name: {w: {k: value[w]['week'][k] for k in KEYS if k in value[w]['week']}
                              for w in ('dry', 'rain')}
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
            for mode, method in present(rows) for seed in SEEDS
            if (mode, method, seed) in rows}}
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
