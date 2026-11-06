"""e06의 4방법 비교를 같은 seed끼리 집계하고 완료 여부를 점검한다."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

CODE = Path(__file__).resolve().parents[2] / 'code/wwtp'
sys.path.insert(0, str(CODE))
from ppo_tail import empirical_cvar

METHODS = ('ppolag', 'ppolag_tail', 'ppolag_fcsrl', 'ppolag_fcsrl_tail')
LABELS = {'ppolag': 'PPO-Lag', 'ppolag_tail': 'PPO-Lag + tail',
          'ppolag_fcsrl': 'FCSRL + PPO-Lag', 'ppolag_fcsrl_tail': 'FCSRL + PPO-Lag + tail'}


def read(path):
    return json.loads(path.read_text())


def summarize(root, output):
    status, manifest = read(root / 'status.json'), read(root / 'manifest.json')
    assert status['phase'] == 'complete', 'Runs are not complete'
    assert len(status['completed']) == 8 and all(x['exit_code'] == 0 for x in status['completed'])
    source_root = Path(__file__).resolve().parents[3]
    for relative, expected in manifest['source'].items():
        assert hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == expected, relative
    for name, expected in manifest['inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    results = read(root / 'results.json')
    assert len(results) == 8
    assert {(r['method'], r['seed']) for r in results} == {(m, s) for m in METHODS for s in (0, 1)}
    by_key = {(r['method'], r['seed']): r for r in results}
    e05 = {(r['method'], r['seed']): r for r in read(Path(__file__).parent.parent / 'e05/pilot_results.json')}
    audits = []
    for result in results:
        method, seed = result['method'], result['seed']
        run = root / 'runs' / method / f'seed_{seed}'
        log, pid = read(run / 'train_log.json'), read(run / 'pid_trace.json')
        assert result['evaluation_split'] == 'val' and result['steps'] == 404096
        assert result['validation']['episodes'] == 215
        assert result['tail'] == {'tail_fraction': 0.2, 'weight': 0.03, 'cost_scale': 15.0}
        assert result['tail_enabled'] == method.endswith('_tail')
        assert len(log) == len(pid) == 44
        assert len([r for r in log if 'val' in r]) == 5
        assert all(r['tail_episode_count'] == 32 and abs(r['tail_effective_mass'] - 6.4) < 1e-10 for r in log)
        for kind in ('best', 'final'):
            rows = read(run / f'val_{kind}_episodes.json')
            assert len(rows) == len({(r['day'], r['member']) for r in rows}) == 215
            costs = [r['cost'] for r in rows]
            metrics = result[f'val_{kind}']
            assert abs(statistics.mean(costs) - metrics['cost']) < 1e-10
            assert abs(empirical_cvar(costs)[1] - metrics['cost_cvar']) < 1e-10
            if not method.endswith('_tail'):
                for key in ('cost', 'reward', 'air_m3'):
                    assert abs(metrics[key] - e05[(method, seed)][f'val_{kind}'][key]) < 1e-10, (method, seed, key)
        audits.append({'method': method, 'seed': seed,
            'final_lambda': pid[-1]['lambda'], 'maximum_lambda': max(x['lambda'] for x in pid),
            'maximum_encoder_sync_kl': max(x.get('encoder_sync_kl', 0) for x in log),
            'maximum_tail_to_base_rms': max(x['tail_to_base_rms'] for x in log),
            'maximum_training_cost_cvar': max(x['train_cost_cvar'] for x in log),
            'updates': len(log), 'elapsed_seconds': result['elapsed_seconds']})
    summary = {'launch_commit': manifest['commit'], 'completed_at': status['updated_at'],
               'seeds': [0, 1], 'split': 'val', 'report_limit': 15, 'tail_fraction': 0.2,
               'baseline_metrics_match_e05': True, 'audits': audits, 'methods': {}, 'paired_deltas': {}}
    lines = ['# Yoo/e06 하루 비용 CVaR 확장 후보의 시험 결과', '',
        '개인 탐색 실험. seed 2개, 실행당 404,096전이, 검증 43일 x 구성원 5개. test는 사용하지 않았다.', '',
        '평균 ± 표준편차는 두 학습 seed 사이의 값이다. 날짜나 대리 모델 구성원을 독립적인 학습 seed로 세지 않는다. CVaR20은 각 정책의 검증 하루 비용 중 상위 20% 평균이며, 215 episode에서 tail 질량은 43개다.', '']
    for key, title in (('val_best', '동일 기준 선택 모델'), ('val_final', '마지막 모델')):
        lines += [f'## {title}', '', '| 방법 | 공기량 (Nm3/일) | 하루 평균 비용 | CVaR20 | 평균 비용 15 이하 seed |',
                  '| --- | --- | --- | --- | --- |']
        for method in METHODS:
            values = [by_key[(method, seed)][key] for seed in (0, 1)]
            stats = {'feasible_seeds': sum(v['cost'] <= 15 for v in values)}
            for field in ('air_m3', 'cost', 'cost_cvar'):
                vals = [v[field] for v in values]
                stats[f'{field}_mean'], stats[f'{field}_sd'] = statistics.mean(vals), statistics.stdev(vals)
            summary['methods'].setdefault(method, {})[key] = stats
            lines.append(f'| {LABELS[method]} | {stats["air_m3_mean"]:,.0f} ± {stats["air_m3_sd"]:,.0f} | '
                f'{stats["cost_mean"]:.2f} ± {stats["cost_sd"]:.2f} | '
                f'{stats["cost_cvar_mean"]:.2f} ± {stats["cost_cvar_sd"]:.2f} | {stats["feasible_seeds"]}/2 |')
        lines += ['', 'tail 추가에 따른 동일 seed의 차이(추가 있음에서 추가 없음을 뺀 값). 음수는 감소를 뜻한다.', '',
                  '| 기본 구성 | seed | 공기량 차이 (Nm3/일) | 평균 비용 차이 | CVaR20 차이 |',
                  '| --- | --- | --- | --- | --- |']
        pairs = []
        for base in ('ppolag', 'ppolag_fcsrl'):
            for seed in (0, 1):
                before, after = by_key[(base, seed)][key], by_key[(base + '_tail', seed)][key]
                delta = {field: after[field] - before[field] for field in ('air_m3', 'cost', 'cost_cvar')}
                pairs.append({'base': base, 'seed': seed, **delta})
                lines.append(f'| {LABELS[base]} | {seed} | {delta["air_m3"]:+,.0f} | '
                             f'{delta["cost"]:+.3f} | {delta["cost_cvar"]:+.3f} |')
        summary['paired_deltas'][key] = pairs
        lines.append('')
    lines += ['## 개별 실행', '',
        '| 방법 | seed | 선택 epoch | 선택 비용 | 선택 CVaR20 | 선택 공기량 | 마지막 비용 | 마지막 CVaR20 | 마지막 공기량 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for result in results:
        best, final = result['val_best'], result['val_final']
        lines.append(f'| {LABELS[result["method"]]} | {result["seed"]} | {result["best_epoch"]} | '
            f'{best["cost"]:.3f} | {best["cost_cvar"]:.3f} | {best["air_m3"]:,.0f} | '
            f'{final["cost"]:.3f} | {final["cost_cvar"]:.3f} | {final["air_m3"]:,.0f} |')
    lines += ['', '## 검증과 해석 범위', '',
        '8개 실행의 정상 종료, 44회 갱신과 5회 고정 검증, 각 rollout의 완결된 하루 32개, 입력과 실행 소스의 SHA256을 확인했다. 위험 항이 없는 두 비교군의 선택 및 마지막 정책 지표가 e05와 정확히 일치한다.', '',
        '선택은 평균 비용 15 이하인 시점 중 최고 보상 기준이다. 해당 시점이 없으면 최소 비용을 선택하고 실패로 표시한다. CVaR로 모델을 선택하지 않았다. 가중치는 0.03으로 고정했으며 결과에 따라 재조정하지 않았다.', '',
        'CVaR 감소와 공기량 증가의 교환관계, seed별 방향을 함께 해석해야 한다. seed 2개와 기존 검증 자료의 탐색 결과로 유의성이나 실제 공정 안전성을 주장하지 않는다. PPO 전환은 기존 기법 적용이고, CVaR 항 추가가 이번 확장 후보다.', '',
        f'실행 커밋: `{manifest["commit"]}`. 원본 FCSRL 커밋: `{manifest["upstream_commit"]}`.', '',
        f'완료 시각(UTC): `{status["updated_at"]}`. 전체 로그와 모델: `{root}`.', '']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.md').write_text('\n'.join(lines))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output / 'pilot_results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs, args.out)
