"""e21의 고정 설정값 보간 오차와 규칙 결정 주기를 별도로 검증한다."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e18'))
from setpoint_env import SetpointEnv, HoldPolicy, setpoint_to_action, episode_metrics
from abac import FeedForward

ENERGY = 'aeration_kwh_per_day'
NH4 = 'flow_weighted_nh4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(prepared, weather, sp=None, interval=1):
    initial = dict(np.load(Path(prepared) / 'initial.npz'))
    meta = json.loads((Path(prepared) / 'metadata.json').read_text())
    if sp is None:
        selected = meta['feedforward_selected']
        policy = HoldPolicy(FeedForward(selected['a'], selected['b'], meta['load_reference_l0']), interval)
    else:
        policy = lambda o, e: [setpoint_to_action(sp)]
    env = SetpointEnv(initial, weather=weather, days=14., tanks=tuple(meta['tanks']))
    obs, _ = env.reset()
    infos = []
    for _ in range(env.T):
        obs, _, _, _, info = env.step(np.asarray(policy(obs, env), dtype=float).reshape(1))
        if env.t > 7 * 288:
            infos.append(info)
    return episode_metrics(infos)


def match_constant(job):
    prepared, weather, name, target, frontier = job
    points = sorted((float(s), x[weather]) for s, x in frontier.items())
    if not all(a[1][NH4] > b[1][NH4] for a, b in zip(points, points[1:])):
        raise ValueError('Constant frontier is not strictly monotone')
    brackets = [(a, b) for a, b in zip(points, points[1:]) if b[1][NH4] <= target[NH4] <= a[1][NH4]]
    if not brackets:
        raise ValueError(f'{weather}/{name}: no bracket, extrapolation forbidden')
    (lo, ml), (hi, mh) = brackets[0]
    interpolated_energy = float(np.interp(target[NH4], [mh[NH4], ml[NH4]], [mh[ENERGY], ml[ENERGY]]))
    cache = {lo: ml, hi: mh}
    evaluations = []

    def objective(sp):
        if sp not in cache:
            cache[sp] = replay(prepared, weather, sp=sp)
            evaluations.append({'sp': sp, NH4: cache[sp][NH4], ENERGY: cache[sp][ENERGY]})
        return cache[sp][NH4] - target[NH4]

    root = float(brentq(objective, lo, hi, xtol=1e-6, maxiter=30))
    # Even an endpoint solution must be directly replayed, rather than copied from the old table.
    actual = replay(prepared, weather, sp=root) if root in (lo, hi) else cache[root]
    mismatch = actual[NH4] - target[NH4]
    if abs(mismatch) > 1e-4:
        raise AssertionError(f'NH4 match failed: {mismatch}')
    old_rel = 100 * (target[ENERGY] / interpolated_energy - 1)
    direct_rel = 100 * (target[ENERGY] / actual[ENERGY] - 1)
    return {'name': name, 'weather': weather, 'target': target, 'matched_sp': root,
            'constant': actual, 'nh4_mismatch': mismatch,
            'interpolated_energy': interpolated_energy, 'old_rel_percent': old_rel,
            'direct_rel_percent': direct_rel, 'correction_pp': direct_rel - old_rel,
            'bracket': [lo, hi], 'evaluations': evaluations}


def parallel(jobs, fn, workers, label):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *j) if fn is replay else pool.submit(fn, j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            print(f'{label} {len(rows)}/{len(jobs)}', flush=True)
    return rows


def ci95(values):
    x = np.asarray(values)
    half = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - half), float(x.mean() + half)]


def report(rows, checks, out):
    lines = ['# e22 비교 기준 검증 결과', '',
             '음수는 같은 평균 NH4를 내는 고정 설정값보다 폭기 에너지가 적다는 뜻이다. '
             '규칙의 파라미터와 e21 정책은 고정했다. 수질을 맞춘 고정 설정값은 사후 비교용이며 운전 정책으로 선택하지 않는다.', '']
    summary = {}
    for weather in ('dry', 'rain'):
        byname = {r['name']: r for r in rows if r['weather'] == weather}
        lines += [f'## {weather}', '', '| 정책 | NH4 | 에너지 kWh/일 | 대응 고정 DO | 기존 보간 대비 % | 직접 계산 대비 % | 보정 %p |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for name, r in sorted(byname.items()):
            lines.append(f"| {name} | {r['target'][NH4]:.6f} | {r['target'][ENERGY]:.2f} | {r['matched_sp']:.6f} | "
                         f"{r['old_rel_percent']:+.4f} | {r['direct_rel_percent']:+.4f} | {r['correction_pp']:+.4f} |")
        summary[weather] = {}
        for method in ('ppo', 'fcsrl'):
            group = [byname[f'{method}_{s}'] for s in range(5)]
            vals = [r['direct_rel_percent'] for r in group]
            ci = ci95(vals)
            summary[weather][method] = {'mean_rel_percent': float(np.mean(vals)), 'ci95': ci,
                'mean_correction_pp': float(np.mean([r['correction_pp'] for r in group])),
                'gap_to_ff30_pp': float(np.mean(vals) - byname['feedforward_30min']['direct_rel_percent'])}
            lines += ['', f"{method}: 직접 계산 대비 {np.mean(vals):+.4f}%, seed 5개 t 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]."]
        pvalue = stats.ttest_ind([byname[f'fcsrl_{s}']['direct_rel_percent'] for s in range(5)],
                                [byname[f'ppo_{s}']['direct_rel_percent'] for s in range(5)], equal_var=False).pvalue
        summary[weather]['welch_rel_p'] = float(pvalue)
        summary[weather]['ff_5min_rel'] = byname['feedforward_5min']['direct_rel_percent']
        summary[weather]['ff_30min_rel'] = byname['feedforward_30min']['direct_rel_percent']
        lines += ['', f'FCSRL 대 PPO 효율 Welch p = {pvalue:.6f}. 비유의 결과는 동등성 입증이 아니다.', '']
    summary['max_nh4_match_error'] = max(abs(r['nh4_mismatch']) for r in rows)
    summary['replay_checks'] = checks
    lines += ['## 검증 범위', '',
              f"수질 일치 최대 오차: {summary['max_nh4_match_error']:.3g} mg N/L. 외삽 없음.",
              '기존 고정 DO 2.0 및 5분 규칙을 두 날씨에서 다시 실행해 저장 지표와 대조했다.',
              '건기가 주 평가이고 우기는 이미 여러 실험에서 확인한 보조 평가다. 독립 최종 test로 해석하지 않는다.',
              'CI는 고정된 날씨에서 학습 seed 간 편차를 나타내며 공정이나 날씨 전체의 불확실성은 아니다.',
              '정책 자체의 수질과 에너지 지표는 바뀌지 않았다. 수정 대상은 비교 기준이다.', '']
    (out / 'results.md').write_text('\n'.join(lines))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.runs.glob('*/seed_*/metrics.json'))
    if len(files) != 10:
        raise ValueError('Expected exactly ten e21 policies')
    inputs = files + [a.prepared / s for s in ('initial.npz', 'metadata.json', 'eval_frontier.json', 'references.json')]
    source_files = [Path(__file__).resolve(), HERE.parent / 'e18/setpoint_env.py', HERE.parent / 'e18/abac.py',
                    HERE.parent / 'e17/bsm_mean.py', HERE.parent / 'e16/bsm_env.py']
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                'input_sha256': {str(f): sha(f) for f in inputs},
                'source_sha256': {str(f): sha(f) for f in source_files},
                'argv': sys.argv, 'python': sys.version}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    frontier = json.loads((a.prepared / 'eval_frontier.json').read_text())
    refs = json.loads((a.prepared / 'references.json').read_text())
    checks, ff = {}, {}
    # Explicit keys keep asynchronous completion order from mixing weather or policies.
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(replay, str(a.prepared), w, sp, k): (w, name)
                   for w in ('dry', 'rain') for name, sp, k in
                   [('constant_2', 2., 1), ('feedforward_5min', None, 1), ('feedforward_30min', None, 6)]}
        for future in as_completed(futures):
            w, name = futures[future]
            actual = future.result()
            ff[w, name] = actual
            if name != 'feedforward_30min':
                expected = frontier['2.0'][w] if name == 'constant_2' else refs['feedforward'][w]['week']
                delta = {k: actual[k] - expected[k] for k in (ENERGY, NH4, 'nh4_violation_percent')}
                if any(abs(x) > 1e-7 for x in delta.values()):
                    raise AssertionError(f'Replay mismatch {w}/{name}: {delta}')
                checks[f'{w}/{name}'] = delta
            print(f'replay verified {w}/{name}', flush=True)
    jobs = []
    seen = set()
    for f in files:
        m = json.loads(f.read_text())
        identity = (m['method'], m['seed'])
        if identity in seen or m['method'] not in ('ppo', 'fcsrl') or m['seed'] not in range(5):
            raise ValueError(f'Invalid identity {identity}')
        seen.add(identity)
        for w in ('dry', 'rain'):
            jobs.append((str(a.prepared), w, f"{m['method']}_{m['seed']}", m['final'][w]['week'], frontier))
    for w in ('dry', 'rain'):
        for name in ('feedforward_5min', 'feedforward_30min'):
            jobs.append((str(a.prepared), w, name, ff[w, name], frontier))
    rows = parallel(jobs, match_constant, a.workers, 'matched')
    rows.sort(key=lambda r: (r['weather'], r['name']))
    for f in inputs:
        if sha(f) != manifest['input_sha256'][str(f)]:
            raise AssertionError(f'Input changed during audit: {f}')
    (a.out / 'matches.json').write_text(json.dumps(rows, indent=2) + '\n')
    report(rows, checks, a.out)
    print(f'Completed in {time.monotonic() - start:.1f}s: {a.out}', flush=True)


if __name__ == '__main__':
    main()
