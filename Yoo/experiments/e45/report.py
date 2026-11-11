"""저장한 rollout을 독립 재집계하여 날짜와 행동 방식의 효과를 비교한다."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

PRIMARY = ('air_m3', 'nh4_cost', 'do_cost')


def main(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'complete' and status['completed'] == 96
    assert hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest() == status['protocol_sha256']
    for path, digest in protocol['input_and_source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    groups, checks, replays = defaultdict(list), 0, 0
    for job in protocol['jobs']:
        path = root / 'jobs' / job['id']
        item = json.loads((path / 'metrics.json').read_text())
        rows = json.loads((path / 'episodes.json').read_text())
        expected = [(day, m) for day in protocol['dates'][job['split']] for m in range(5)]
        assert [(r['day'], r['member']) for r in rows] == expected
        assert item['frozen_policy']
        if item['e44_reproduced']:
            replays += 1
        with np.load(path / 'rollout.npz') as z:
            states, air = z['states'], z['air']
        assert states.shape == (len(expected), 95, 5) and air.shape == (len(expected), 95)
        assert np.isfinite(states).all() and np.isfinite(air).all()
        values = dict(air_m3=np.sum(air, axis=1) / 4,
                      nh4_cost=np.count_nonzero(states[:, :, 0] > 4, axis=1),
                      do_cost=np.count_nonzero(np.any(states[:, :, 2:] > 2, axis=2), axis=1),
                      nh4_excess_mg_h_L=np.clip(states[:, :, 0] - 4, 0, None).sum(1) / 4,
                      do_excess_mg_h_L=np.clip(states[:, :, 2:].max(2) - 2, 0, None).sum(1) / 4)
        for key, val in values.items():
            np.testing.assert_allclose([r[key] for r in rows], val, rtol=1e-10, atol=1e-10)
            assert np.isclose(val.mean(), item['summary'][key], rtol=1e-10, atol=1e-10)
            for member in range(5):
                assert np.isclose(val[member::5].mean(), item['members'][str(member)][key])
            checks += 7
        for name in ('nh4_cost', 'do_cost'):
            assert np.isclose(values[name].reshape(-1, 5).mean(1).max(), item['summary']['worst_day_' + name])
            checks += 1
        key = (job['policy'], job['split'], 'deterministic' if job['repeat'] is None else 'stochastic')
        groups[key].append(item)
    assert replays == 12
    aggregated = []
    for (policy, split, action), items in groups.items():
        assert len(items) == (1 if action == 'deterministic' else 3)
        keys = list(values)
        row = dict(policy=policy, split=split, action=action, repeats=len(items),
                   mean={k: float(np.mean([x['summary'][k] for x in items])) for k in keys},
                   repeat_range={k: [min(x['summary'][k] for x in items), max(x['summary'][k] for x in items)] for k in keys},
                   mean_worst_day={k: float(np.mean([x['summary']['worst_day_' + k] for x in items])) for k in ('nh4_cost', 'do_cost')},
                   members={str(m): {k: float(np.mean([x['members'][str(m)][k] for x in items])) for k in keys} for m in range(5)})
        row['both_mean_limits'] = row['mean']['nh4_cost'] <= 6 and row['mean']['do_cost'] <= 6
        aggregated.append(row)
    lookup = {(x['policy'], x['split'], x['action']): x for x in aggregated}
    contrasts = []
    lines = ['# e45 동일 조건 평가 결과', '',
             '전체 train188일과 개발 val40일, 모델5개, 고정 정책12개. 확률 행동은3회 평균이며 독립 정책 seed가 아니다.',
             '새 학습과 test 평가는 없다. 수치는 대리 환경 내부 결과다.', '',
             '## NH4 비용', '',
             '| 정책 | train 결정적 | train 확률적 | val 결정적 | val 확률적 |',
             '| --- | --- | --- | --- | --- |']
    for pol in protocol['policies']:
        pid = pol['id']
        td, ts, vd, vs = [lookup[(pid, split, action)] for split, action in
                         [('train', 'deterministic'), ('train', 'stochastic'), ('val', 'deterministic'), ('val', 'stochastic')]]
        lines.append('| ' + pid + ' | ' + ' | '.join(f"{x['mean']['nh4_cost']:.3f}" for x in (td, ts, vd, vs)) + ' |')
        contrasts.append(dict(policy=pid,
            val_minus_train={a: {k: lookup[(pid, 'val', a)]['mean'][k] - lookup[(pid, 'train', a)]['mean'][k] for k in PRIMARY} for a in ('deterministic', 'stochastic')},
            stochastic_minus_deterministic={s: {k: lookup[(pid, s, 'stochastic')]['mean'][k] - lookup[(pid, s, 'deterministic')]['mean'][k] for k in PRIMARY} for s in ('train', 'val')},
            nh4_failure_all_four=all(x['mean']['nh4_cost'] > 6 for x in (td, ts, vd, vs))))
    lines += ['', '## 세 주 지표와 초과 적분', '',
              '| 정책 | 날짜 | 행동 | 공기 m³/23.75h | NH4 비용 | DO 비용 | NH4 초과 적분 | DO 초과 적분 | 두 평균6/6 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in aggregated:
        v = row['mean']
        lines.append(f"| {row['policy']} | {row['split']} | {row['action']} | {v['air_m3']:.1f} | {v['nh4_cost']:.3f} | {v['do_cost']:.3f} | {v['nh4_excess_mg_h_L']:.3f} | {v['do_excess_mg_h_L']:.3f} | {row['both_mean_limits']} |")
    count = sum(x['nh4_failure_all_four'] for x in contrasts)
    lines += ['', f'네 조건 모두 NH4 평균6을 초과한 정책은 {count}/12개다. 이는 표현 head 분리가 해결책이라는 증명은 아니다.',
              f'기존 e44 결정적 val 재현 {replays}/12개, 독립 수치 검사 {checks}개 통과.',
              'report.json에 반복별 범위, 모델별 결과, 반복별 최악 날짜의 평균과 조건 간 차이를 보관한다. 최악 날짜 평균은 모든 반복 중 최댓값과 다르다.',
              '날짜 분할은 무작위 배정이 아니므로 분포 변화의 인과 효과로 단정하지 않는다. 개발 자료는 반복 사용했고 환경 오차는 남는다.']
    report = dict(evaluation_code_commit=protocol['code_commit'], protocol_sha256=status['protocol_sha256'],
                  report_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=status['elapsed_seconds'], independent_checks=checks, e44_replays=replays,
                  aggregated=aggregated, contrasts=contrasts, nh4_failure_all_four_count=count,
                  test_evaluated=False, weight_updates=0, adopted=False)
    (root / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('REPORT COMPLETE', checks, 'checks;', replays, 'replays;', count, 'fail all four', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        while True:
            state = json.loads((args.root / 'status.json').read_text())
            if state['phase'] == 'failed':
                raise RuntimeError(state['error'])
            if state['phase'] == 'complete':
                break
            time.sleep(5)
    main(args.root)
