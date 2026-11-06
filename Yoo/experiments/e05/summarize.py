"""완료된 e05 시험의 선택 모델과 마지막 모델을 별도로 집계한다."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

METHODS = ('ppolag', 'ppolag_fcsrl', 'ppolag_frozen')
LABELS = {'ppolag': 'PPO-Lag', 'ppolag_fcsrl': 'FCSRL + PPO-Lag',
          'ppolag_frozen': '표현 고정 PPO-Lag'}


def summarize(root, output):
    status = json.loads((root / 'status.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert status['phase'] == 'complete', 'Runs are not complete'
    assert len(status['completed']) == 6 and all(x['exit_code'] == 0 for x in status['completed'])
    source_root = Path(__file__).resolve().parents[3]
    for relative, expected in manifest['source'].items():
        assert hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == expected, relative
    results = json.loads((root / 'results.json').read_text())
    assert {(r['method'], r['seed']) for r in results} == {(m, s) for m in METHODS for s in (0, 1)}
    audits = []
    for result in results:
        run = root / 'runs' / result['method'] / f'seed_{result["seed"]}'
        log = json.loads((run / 'train_log.json').read_text())
        pid = json.loads((run / 'pid_trace.json').read_text())
        assert result['evaluation_split'] == 'val' and result['steps'] == 404096
        assert result['validation']['episodes'] == 215
        assert len(log) == len(pid) == 44
        assert len([r for r in log if 'val' in r]) == 5
        for kind in ('best', 'final'):
            rows = json.loads((run / f'val_{kind}_episodes.json').read_text())
            assert len(rows) == len({(r['day'], r['member']) for r in rows}) == 215
            assert abs(statistics.mean(r['cost'] for r in rows) - result[f'val_{kind}']['cost']) < 1e-10
        audits.append({'method': result['method'], 'seed': result['seed'],
            'final_lambda': pid[-1]['lambda'], 'maximum_lambda': max(x['lambda'] for x in pid),
            'maximum_encoder_sync_kl': max(x.get('encoder_sync_kl', 0) for x in log),
            'updates': len(log), 'elapsed_seconds': result['elapsed_seconds']})
    summary = {'launch_commit': manifest['commit'], 'completed_at': status['updated_at'],
               'seeds': [0, 1], 'split': 'val', 'report_limit': 15, 'audits': audits, 'methods': {}}
    lines = ['# Yoo/e05 PPO 기반 FCSRL 시험 결과', '',
        '개인 탐색 실험. seed 2개, 실행당 404,096전이, 검증 43일 x 구성원 5개. test는 사용하지 않았다.', '',
        '평균 ± 표준편차는 두 학습 seed 사이의 값이다. 대리 모델 구성원이나 날짜를 독립적인 학습 seed로 세지 않는다.', '']
    for key, title in (('val_best', '동일 기준 선택 모델'), ('val_final', '마지막 모델')):
        lines += [f'## {title}', '', '| 방법 | 공기량 (Nm3/일) | 하루 비용 | 비용 15 이하 seed |',
                  '| --- | --- | --- | --- |']
        for method in METHODS:
            values = [r[key] for r in results if r['method'] == method]
            air = [r['air_m3'] for r in values]
            cost = [r['cost'] for r in values]
            stats = {'air_mean': statistics.mean(air), 'air_sd': statistics.stdev(air),
                'cost_mean': statistics.mean(cost), 'cost_sd': statistics.stdev(cost),
                'feasible_seeds': sum(c <= 15 for c in cost)}
            summary['methods'].setdefault(method, {})[key] = stats
            lines.append(f'| {LABELS[method]} | {stats["air_mean"]:,.0f} ± {stats["air_sd"]:,.0f} | '
                         f'{stats["cost_mean"]:.2f} ± {stats["cost_sd"]:.2f} | {stats["feasible_seeds"]}/2 |')
        lines.append('')
    lines += ['## 개별 실행', '', '| 방법 | seed | 선택 epoch | 선택 비용 | 선택 공기량 | 마지막 비용 | 마지막 공기량 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for result in results:
        best, final = result['val_best'], result['val_final']
        lines.append(f'| {LABELS[result["method"]]} | {result["seed"]} | {result["best_epoch"]} | '
                     f'{best["cost"]:.3f} | {best["air_m3"]:,.0f} | {final["cost"]:.3f} | {final["air_m3"]:,.0f} |')
    lines += ['', '## 범위와 해석', '',
        '선택은 검증 평균 비용 15 이하인 평가 시점 중 최고 보상 기준이다. 만족 모델이 없으면 최소 비용을 고르고 실패 표시를 유지한다. 마지막 모델은 별도로 보고한다.', '',
        'PPO 기반 FCSRL은 원본 논문에 이미 존재한다. 이번 결과를 새로운 확장 기법의 성공으로 부르지 않는다. 표현 고정 대조는 구조 차이와 표현 학습의 효과를 구분하기 위한 것이다.', '',
        'seed 2개와 기존 검증 자료의 탐색 결과다. 유의성, 최종 수렴 또는 실제 공정 안전성을 주장하지 않는다. 원본 대비 구현 및 환경 차이는 README에 기록했다.', '',
        f'실행 커밋: `{manifest["commit"]}`. 원본 FCSRL 커밋: `{manifest["upstream_commit"]}`.', '',
        f'런타임 결과 위치: `{root}`.', '']
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
