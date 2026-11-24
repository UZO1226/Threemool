"""Recompute daily effects from saved branches and write the seed results."""
import argparse
import json
from pathlib import Path

import numpy as np


def passes(row):
    return (row[..., 0] < -1e-8) & (row[..., 1:] <= 1e-8).all(axis=-1)


def report(root, dest):
    records = json.loads((root / 'summary.json').read_text())
    assert [r['seed'] for r in records] == list(range(5))
    lines = ['# e94 첫 감량 후 두 번째 감량의 효과', '',
             '차이는 선택 방식 빼기 비교 기준이다. 공기는 m³/6시간 창, 두 위반은 15분 스텝, NH4 초과 적분은 mg h/L다.', '']
    for group, title in [('supported', '거리 조건 안'), ('all', '모든 비중복 후보')]:
        selected = []
        direct = []
        saved_and_protected = 0
        net_air_increases = 0
        for rec in records:
            seed = rec['seed']
            result = rec['groups'][group]
            with np.load(root / f'pulse{seed}.npz') as z:
                mask = z['supported'] if group == 'supported' else np.ones(len(z['keys']), bool)
                first = z['first'][mask]
                possible = first & z['eligible'][mask]
                take = possible & (z['level'][mask] <= 4) & (z['trend'][mask] <= 0)
                np.testing.assert_array_equal(take, first & z['rule'][mask])
                day = z['keys'][mask, 0]
                a, b, c = (z[k][mask] for k in ('a', 'b', 'c'))
                first_total = np.where(first[:, None], b, a)
                best = possible & passes(c-b)
                direct.append(-62.5 * np.mean([take[day == d].mean() for d in np.unique(day)]))
                saved_and_protected += int((best & take).sum())
                net_air_increases += int((take & (c[:, 0] - b[:, 0] > 1e-8)).sum())
                expected = []
                for date in np.unique(day):
                    use = day == date
                    eligible = use & possible
                    random_effect = np.zeros(4)
                    if eligible.any():
                        random_effect = (c-b)[eligible].mean(0) * take[use].sum() / use.sum()
                    expected.append(random_effect)
                for mode, decision in [('first_only', np.zeros(len(day), bool)), ('nh_rule', take),
                                       ('all_possible', possible), ('perfect', best), ('matched_random', take)]:
                    totals = np.where(decision[:, None], c, first_total)
                    extra = np.mean(expected, axis=0) if mode == 'matched_random' else np.mean(
                        [(totals-first_total)[day == d].mean(0) for d in np.unique(day)], axis=0)
                    base = np.mean([(first_total-a)[day == d].mean(0) for d in np.unique(day)], axis=0)
                    stats = result['modes'][mode]
                    np.testing.assert_allclose([extra, base+extra], [stats['vs_first'], stats['vs_base']],
                                               atol=1e-7, rtol=0)
                    np.testing.assert_allclose(decision.sum(), stats['second'])
            selected.append(result)
        lines += [f'## {title}', '', '| 비교 기준 | 방식 | 후속 감량 수 | 공기 차이 | NH4 위반 차이 | DO 위반 차이 | NH4 초과 차이 | 공동 개선 seed |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for key, label in [('vs_base', '기본 FCSRL'), ('vs_first', '첫 감량만')]:
            for mode, name in [('first_only', '첫 감량만'), ('nh_rule', 'NH4 규칙'), ('matched_random', '날짜별 무작위'),
                               ('all_possible', '공기 범위 안 전부'), ('perfect', '사후 선택')]:
                stats = [g['modes'][mode] for g in selected]
                values = np.array([s[key] for s in stats])
                count = sum(s['second'] for s in stats)
                fmt = ' | '.join(f'{v:+.5f}' for v in values.mean(0))
                lines.append(f'| {label} | {name} | {count:.0f} | {fmt} | {passes(values).sum()}/5 |')
        lines += ['', '| seed | 전체 | 첫 감량 | 후속 가능 | NH4 규칙 실행 | 첫 감량만 대비 공기 | NH4 위반 | DO 위반 | NH4 초과 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed, g in enumerate(selected):
            r = g['modes']['nh_rule']
            fmt = ' | '.join(f'{v:+.5f}' for v in r['vs_first'])
            lines.append(f"| {seed} | {g['n']} | {g['first']} | {g['possible']} | {r['second']:.0f} | {fmt} |")
        net = np.mean([g['modes']['nh_rule']['vs_first'][0] for g in selected])
        lines += ['', f'사후 분해에서 즉시 공기 차이는 {np.mean(direct):+.2f} m³, 이후 되먹임 차이는 {net-np.mean(direct):+.2f} m³다. '
                  f'실행 창 중 공동 개선은 {saved_and_protected}건, 총공기 증가는 {net_air_increases}건이다.', '']
    dest.write_text('\n'.join(lines).rstrip() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.dest)
