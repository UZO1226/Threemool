"""Audit saved choices and report the role of the first and second cuts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from second_bound import SCOPES, bounds


def export(root, dest):
    records = json.loads((root / 'e95/main/certificates.json').read_text())
    lines = ['# e95 기본 FCSRL 대비 공기 절감 상한', '',
             '공기 차이는 선택 경로 빼기 기본 FCSRL이며 음수가 절감이다. 공기 단위는 m³/6시간 창이다.', '']
    compact = []
    for record in records:
        path = root / 'e94/main' / f"pulse{record['seed']}.npz"
        assert sha256(path.read_bytes()).hexdigest() == record['sha256']
        results = {}
        with np.load(path) as z:
            for name, g in record['groups'].items():
                mask = z['supported'] if name == 'supported' else np.ones(len(z['keys']), bool)
                day = z['keys'][mask, 0]
                a, b, c = (z[k][mask] for k in ('a', 'b', 'c'))
                first = z['first'][mask]
                possible = first & z['eligible'][mask]
                both = np.where(possible[:, None], c, np.where(first[:, None], b, a))
                once = np.where(first[:, None], b, a)
                neither_safe = ((once[:, 1:] > a[:, 1:]).any(1) & (both[:, 1:] > a[:, 1:]).any(1))
                np.testing.assert_array_equal(np.flatnonzero(neither_safe), g['bad_rows'])
                v = {k:g[k] for k in ('n','days','first','possible','bad_no_second','baseline_air')}
                v['bad_windows'] = int(neither_safe.sum())
                for family in ('fixed', 'optional'):
                    v[family] = {}
                    for scope in SCOPES:
                        p = g[family][scope]
                        summary = {k:p[k] for k in ('status','method','delta','savings_pct','floors','second',
                                                     'worse_days','changed_first') if k in p}
                        if scope == 'date':
                            summary['bad_dates'] = len(p.get('failed_dates', []))
                        if p['status'] == 'optimal':
                            ids = np.array(p['selected'])
                            assert family == 'optional' or not (ids == 0).any()
                            outcome = np.where((ids == 2)[:, None], both, np.where((ids == 1)[:, None], once, a)) - a
                            days = np.stack([outcome[day == d].mean(0) for d in np.unique(day)])
                            np.testing.assert_allclose(days.mean(0), p['delta'], atol=1e-8, rtol=0)
                            quality = dict(pooled=days.mean(0)[None], date=days, window=outcome)[scope]
                            assert (quality[:, 1:] <= bounds.TOL).all()
                        v[family][scope] = summary
                # With per-window quality protection, comparing two or three safe options is exact.
                local = np.where((once[:, 1:] <= a[:, 1:]).all(1), np.minimum(a[:, 0], once[:, 0]), a[:, 0])
                full = np.where((both[:, 1:] <= a[:, 1:]).all(1), np.minimum(local, both[:, 0]), local)
                w = 1 / np.bincount(day)[day]
                v['first_choice_alone_m3'] = float(np.average(a[:, 0]-local, weights=w))
                v['extra_second_choice_m3'] = float(np.average(local-full, weights=w))
                np.testing.assert_allclose(np.average(full-a[:, 0], weights=w),
                                           v['optional']['window']['delta'][0], atol=1e-8)
                results[name] = v
        compact.append(dict(seed=record['seed'], sha256=record['sha256'], groups=results))
    for group, caption in [('supported', '거리 조건 안'), ('all', '전체 비중복 창')]:
        rows = [r['groups'][group] for r in compact]
        lines += [f'## {caption}', '', '첫 감량 결정을 고정한 결과다. NH4 최소 차이는 다른 제약을 모두 풀고 해당 지표만 최소화했다.', '',
                  '| seed | 창 | 첫 감량 | 두 번째 선택 가능 | 기간 최적 공기 차이 | NH4 위반 최소 차이 | 보호 불가 날짜 | 보호 불가 창 (후속 불가 창) |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed, g in enumerate(rows):
            p = g['fixed']['pooled']
            air = f"{p['delta'][0]:+.2f}" if 'delta' in p else '수질 불가'
            lines.append(f"| {seed} | {g['n']} | {g['first']} | {g['possible']} | {air} | {p['floors'][0]:+.5f} | "
                         f"{g['fixed']['date']['bad_dates']} | {g['bad_windows']} ({g['bad_no_second']}) |")
        lines += ['', '기존 첫 감량 대상에서 생략도 허용했을 때의 5개 seed 동일 가중 평균이다.', '',
                  '| 수질 조건 | 최대 절감 | 평균 절감률 | NH4 위반 차이 | DO 위반 차이 | NH4 초과 적분 차이 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for scope, label in zip(SCOPES, ('기간 평균', '날짜마다', '창마다')):
            parts = [g['optional'][scope] for g in rows]
            mean = np.mean([p['delta'] for p in parts], axis=0)
            pct = np.mean([p['savings_pct'] for p in parts])
            lines.append(f'| {label} | {-mean[0]:.2f} | {pct:.2f}% | {mean[1]:+.5f} | {mean[2]:+.5f} | {mean[3]:+.5f} |')
        first_only, second_extra = np.mean([[g['first_choice_alone_m3'], g['extra_second_choice_m3']] for g in rows], axis=0)
        lines += ['', f'사후 분해에서 매 창의 수질을 지키며 최초 감량 여부만 골랐을 때 최대 절감은 {first_only:.2f} m³, '
                  f'그 위에 후속 선택을 더했을 때 추가 가능량은 {second_extra:.2f} m³다.', '']
    (dest / 'summary.json').write_text(json.dumps(compact, indent=1, allow_nan=False) + '\n')
    (dest / 'results.md').write_text('\n'.join(lines).rstrip() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=bounds.ROOT)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.dest)
