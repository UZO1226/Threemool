"""전체 경로 재검산과 사전 지정 seed별 대조."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    records, checks = {}, 0
    lines = ['# e46 결과', '', '2seed 탐색, 대리 환경 내부 결과. 공기는 m³/23.75h, 수질 평균 한도6/6.', '',
             '| 방법 | seed | 날짜 | 공기 | NH4 | DO | 저수온 NH4 | 일반 NH4 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for arm in manifest['arms']:
        for seed in manifest['seeds']:
            path = root / 'runs' / arm / f'seed_{seed}'
            data = json.loads((path / 'metrics.json').read_text())
            assert data['steps'] == 401280 and data['normalizer_and_prior_frozen']
            records[f'{arm}/{seed}'] = data['results']
            for split in ('train', 'val'):
                rows = json.loads((path / f'{split}_episodes.json').read_text())
                with np.load(path / f'{split}_rollout.npz') as z:
                    states, air = z['states'], z['air']
                assert np.isfinite(states).all() and np.isfinite(air).all()
                values = dict(air_m3=air.sum(1) / 4,
                     nh4_cost=np.count_nonzero(states[:, :, 0] > 4, axis=1),
                     do_cost=np.count_nonzero((states[:, :, 2:] > 2).any(2), axis=1),
                     nh4_excess_mg_h_L=np.maximum(states[:, :, 0] - 4, 0).sum(1) / 4,
                     do_excess_mg_h_L=np.maximum(states[:, :, 2:].max(2) - 2, 0).sum(1) / 4)
                r = data['results'][split]
                for key, value in values.items():
                    np.testing.assert_allclose([row[key] for row in rows], value, rtol=1e-10, atol=1e-10)
                    assert np.isclose(value.mean(), r['summary'][key], rtol=1e-10, atol=1e-10)
                    checks += 2
                    for g in (0, 1):
                        mask = np.array([manifest['groups']['group'][row['day']] == g for row in rows])
                        if mask.any():
                            assert np.isclose(value[mask].mean(), r['groups'][str(g)][key])
                            checks += 1
                for key in ('nh4_cost', 'do_cost'):
                    assert np.isclose(values[key].reshape(-1, 5).mean(1).max(), r['summary']['worst_day_' + key])
                    checks += 1
                if arm == 'baseline' and split == 'val':
                    old = Path('/home/jio/fcsrl_src/results/e44/main/runs/fcsrl/padded') / f'seed_{seed}'
                    with np.load(old / 'val_rollout.npz') as z:
                        np.testing.assert_allclose(states, z['states'], atol=1e-6, rtol=0)
                        np.testing.assert_allclose(air, z['air'], atol=1e-3, rtol=0)
                    checks += 2
                s = r['summary']
                lines.append(f"| {arm} | {seed} | {split} | {s['air_m3']:.1f} | {s['nh4_cost']:.3f} | {s['do_cost']:.3f} | {r['groups']['0']['nh4_cost']:.3f} | {r['groups']['1']['nh4_cost']:.3f} |")
    comparisons = []
    lines += ['', '## 사전 지정 탐색 판정', '']
    for seed in manifest['seeds']:
        a, b = records[f'conditional/{seed}'], records[f'baseline/{seed}']
        primary = ('air_m3', 'nh4_cost', 'do_cost')
        guards = ('nh4_excess_mg_h_L', 'do_excess_mg_h_L', 'worst_day_nh4_cost', 'worst_day_do_cost')
        delta = {k: a['val']['summary'][k] - b['val']['summary'][k] for k in primary + guards}
        groups_met = all(a['train']['groups'][str(g)]['nh4_cost'] <= 6 for g in (0, 1))
        passed = all(delta[k] < 0 for k in primary) and all(delta[k] <= 0 for k in guards) and a['val']['summary']['both_mean_limits'] and groups_met
        comparisons.append(dict(seed=seed, val_delta=delta, train_group_constraints_met=groups_met, exploratory_screen=passed))
        lines.append(f'- seed{seed}: 추가 검증 후보 조건 {passed}, train 두 NH4 조건 만족 {groups_met}, val 차이 {delta}')
    lines += ['', '두 seed 모두 통과해도 확증이나 채택은 아니다. 새 조건별 제약은 기존 전체 평균보다 강한 학습 목표다. test 평가는 없다.']
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    (root / 'comparison.json').write_text(json.dumps(dict(independent_checks=checks, results=records,
        comparisons=comparisons, test_evaluated=False, adopted=False), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print('REPORT COMPLETE', checks, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
