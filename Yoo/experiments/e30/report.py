"""고정된 마지막 정책의 seed 평균과 환경 진단을 집계한다. test는 사용하지 않는다."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, ttest_rel
from common import METHODS, save, sha


def main(root, prepared):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['input_sha256'].items():
        assert sha(prepared / name) == digest, name
    summary, rows = {}, []
    import torch
    prior = torch.load(prepared / 'prior.pt', weights_only=True)
    norm = json.loads((prepared / 'normalizer.json').read_text())
    for method in METHODS:
        metrics = []
        for seed in manifest['seeds']:
            run = root / 'runs' / method / f'seed_{seed}'
            result = json.loads((run / 'metrics.json').read_text())
            assert result['steps'] == manifest['steps_per_run'] and result['normalizer_and_prior_frozen']
            assert json.loads((run / 'model_final/normalizer.json').read_text()) == norm
            weights = torch.load(run / 'model_final/prior.pt', weights_only=True)
            assert all(torch.equal(v, weights[k]) for k, v in prior.items())
            metrics.append(result)
            rows.append(result)
        val = [r['val_final'] for r in metrics]
        summary[method] = {k: {'mean': float(np.mean([r[k] for r in val])),
                              'sd': float(np.std([r[k] for r in val], ddof=1)) if len(val)>1 else 0.}
                           for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')}
        summary[method]['seeds_within_limit'] = sum(r['cost'] <= 12 for r in val)
        summary[method]['mean_elapsed_seconds'] = float(np.mean([r['elapsed_seconds'] for r in metrics]))
    # Paired seeds are the replicate unit; 200 surrogate episodes are not 200 independent trained policies.
    contrasts = []
    if len(manifest['seeds']) > 1:
        base = np.array([r['val_final']['cost'] for r in rows if r['method']=='fcsrl'])
        for method in METHODS:
            if method == 'fcsrl':
                continue
            value = np.array([r['val_final']['cost'] for r in rows if r['method']==method])
            delta = value-base
            half = t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            p = float(ttest_rel(value, base).pvalue) if delta.std() > 0 else (1. if np.all(delta==0) else 0.)
            contrasts.append({'method': method, 'cost_delta_vs_fcsrl': float(delta.mean()),
                              'ci95': [float(delta.mean()-half), float(delta.mean()+half)], 'p_raw': p})
        ordered = sorted(contrasts, key=lambda x: x['p_raw'])
        adjusted = 0.
        for i, row in enumerate(ordered):
            adjusted = max(adjusted, min(1., (len(ordered)-i)*row['p_raw']))
            row['p_holm'] = adjusted
    diagnostic = json.loads((prepared.parent / 'environment_diagnostic.json').read_text())
    save(root / 'summary.json', {'methods': summary, 'paired_cost_contrasts': contrasts,
         'runs': rows, 'environment': diagnostic, 'adoption': 'No automatic adoption; surrogate validation only.'})
    md = ['# e30 DARROW 결과', '', f"{len(manifest['seeds'])} seeds, 각 {manifest['steps_per_run']:,} 전이. 마지막 정책, 검증 40일 x 모델 5개.",
          '', '| 방법 | 공기량 Nm³/23.75h | 비용 | 평균 한도 만족 seed |', '| --- | --- | --- | --- |']
    for method, s in summary.items():
        air, cost = s['air_m3'], s['cost']
        md.append(f"| {method} | {air['mean']:.1f} ± {air['sd']:.1f} | {cost['mean']:.2f} ± {cost['sd']:.2f} | {s['seeds_within_limit']}/{len(manifest['seeds'])} |")
    md += ['', f"기록 운전의 관측 비용 {diagnostic['observed_daily_cost']:.2f}, 모델 재생 비용 {diagnostic['model_replay_daily_cost']:.2f}.",
           '모든 값은 95개 전이의 합이다. 실제 공정 안전 또는 에너지 절감 결과가 아니다.',
           '조건 간 공기량 차이가 있으므로 비용 감소만으로 제어 효율 개선을 주장하지 않는다.',
           '추가 같은 공기량 대조와 모델 신뢰도 검토 전 확장을 자동 채택하지 않는다.',
           'Avedore e23~e29 결과와 직접 우열 비교하지 않는다. test와 대회 evaluation 정답은 사용하지 않았다.', '']
    (root / 'results.md').write_text('\n'.join(md))
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.prepared)
