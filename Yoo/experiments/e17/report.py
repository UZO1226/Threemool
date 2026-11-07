"""고정된 8개 시험을 확인하고 초기 및 마지막 정책의 결과를 기록한다."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = {'ppo': 'PPO', 'ppo_tail': 'PPO + CVaR', 'fcsrl': 'FCSRL + PPO', 'fcsrl_tail': 'FCSRL + PPO + CVaR'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.pilot / 'manifest.json').read_text())
    status = json.loads((a.pilot / 'status.json').read_text())
    assert status['phase'] == 'complete' and len(status['completed']) == 8
    for file, sha in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == sha, file
    metadata = json.loads((a.prepared / 'metadata.json').read_text())
    for file, sha in metadata['files'].items():
        assert hashlib.sha256((a.prepared / file).read_bytes()).hexdigest() == sha
    rows = json.loads((a.pilot / 'results.json').read_text())
    refs = json.loads((a.prepared / 'references.json').read_text())
    index = {(r['method'], r['seed']): r for r in rows}
    assert len(index) == 8
    for seed in (0, 1):
        for basic, tail in [('ppo', 'ppo_tail'), ('fcsrl', 'fcsrl_tail')]:
            assert index[basic, seed]['initial'] == index[tail, seed]['initial']
        schedules = [[x['training_days'] for x in index[m, seed]['training']] for m in LABELS]
        assert all(x == schedules[0] for x in schedules)
    for r in rows:
        assert r['steps'] == 23040 and r['updates'] == 4 and len(r['training']) == 4
        assert all(t['policy_parameter_delta_l2'] > 0 for t in r['training'])
        assert all(t['tail_episode_count'] == 20 and t['tail_effective_mass'] == 4 for t in r['training'])
    a.out.mkdir(parents=True, exist_ok=True)
    out = {'manifest': manifest, 'references': refs, 'runs': rows,
           'checks': {'source_unchanged': True, 'prepared_states_unchanged': True,
                      'tail_pairs_same_initial_policy': True, 'paired_day_schedules': True,
                      'eight_complete_runs': True, 'actual_policy_updates': True}}
    (a.out / 'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    lines = ['# e17 제한된 학습 시험 결과', '',
             '4조건 x seed 2개, 각 23,040전이와 4회 갱신이다. 아래 평균과 표준편차는 seed 두 개의 값이며 '
             '독립 최종 test나 수렴 결과가 아니다. 각 정책을 14일 연속 운전하고 후반 7일을 집계했다.', '']
    def stat(xs):
        return f'{np.mean(xs):.3f} ± {np.std(xs, ddof=0):.3f}'
    for weather in ('dry', 'rain'):
        lines += [f'## {weather}', '',
                  '| 방법 | 폭기 kWh/day | NH4 유량 가중 평균 mg N/L | NH4 초과 시간 % | 일별 NH4 CVaR20 |',
                  '| --- | --- | --- | --- | --- |']
        keys = ('aeration_kwh_per_day', 'flow_weighted_nh4', 'nh4_violation_percent', 'daily_nh4_cvar20')
        for name, ref in refs.items():
            lines.append('| ' + name + ' | ' + ' | '.join(f'{ref[weather]["week"][k]:.3f}' for k in keys) + ' |')
        for m in LABELS:
            values = [index[m, s]['final'][weather]['week'] for s in (0, 1)]
            lines.append('| ' + LABELS[m] + ' | ' + ' | '.join(stat([v[k] for v in values]) for k in keys) + ' |')
        lines += ['', '| 방법 | seed | 초기 NH4 | 마지막 NH4 | 초기 폭기 | 마지막 폭기 |',
                  '| --- | --- | --- | --- | --- | --- |']
        for m in LABELS:
            for seed in (0, 1):
                run = index[m, seed]
                before, after = (run[x][weather]['week'] for x in ('initial', 'final'))
                lines.append(f"| {LABELS[m]} | {seed} | {before['flow_weighted_nh4']:.3f} | {after['flow_weighted_nh4']:.3f} | "
                             f"{before['aeration_kwh_per_day']:.1f} | {after['aeration_kwh_per_day']:.1f} |")
        lines += ['']
    lines += ['## 수치와 실행 검사', '',
              '| 방법 | seed | 최종 lambda | 최대 전체 갱신 KL | 평균 경사 clipping % | 최대 행동 clipping % | feasibility 구간 이탈 % |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        tr = r['training']
        lines.append(f"| {LABELS[r['method']]} | {r['seed']} | {tr[-1]['lambda']:.4f} | "
                     f"{max(t['full_update_probe_kl'] for t in tr):.4f} | "
                     f"{100*np.mean([t['gradient_clip_fraction'] for t in tr]):.2f} | "
                     f"{100*max(t['raw_action_clip_fraction'] for t in tr):.2f} | "
                     f"{100*max(t['feasibility_symlog_outside_fraction'] for t in tr):.2f} |")
    lines += ['', '경사 clipping은 actor, critic, encoder 호출을 합친 비율이다. 특정 모듈의 불안정성을 단독으로 뜻하지 않는다. '
              '모든 실행에서 실제 정책 파라미터 변화가 있었고 원본 소스와 준비 상태가 유지됐는지 확인했다.', '',
              f"실행 커밋 `{manifest['code_commit']}`. 시작 {manifest['started_at']}, 완료 {status['updated_at']}. "
              '상세 초기 및 마지막 일별 지표와 원본 기준 제어는 summary.json에 있다.', '',
              '![초기 및 마지막 정책](figures/pilot.png)', '']
    (a.out/'results.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = dict(zip(LABELS, ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:red']))
    for ax, weather in zip(axes, ('dry', 'rain')):
        for m in LABELS:
            for seed in (0, 1):
                run = index[m, seed]
                b, f = (run[x][weather]['week'] for x in ('initial', 'final'))
                x, y = b['aeration_kwh_per_day'], b['flow_weighted_nh4']
                xx, yy = f['aeration_kwh_per_day'], f['flow_weighted_nh4']
                ax.annotate('', xy=(xx, yy), xytext=(x, y), arrowprops={'arrowstyle': '->', 'color': colors[m], 'alpha': .7})
                ax.scatter(xx, yy, color=colors[m], marker='o' if seed == 0 else 's', label=LABELS[m] if seed == 0 else None)
        ref = refs['do_pi'][weather]['week']
        ax.scatter(ref['aeration_kwh_per_day'], ref['flow_weighted_nh4'], color='black', marker='*', s=120, label='DO PI')
        ax.axhline(4, linestyle=':', color='gray')
        ax.set(xlabel='Aeration energy (kWh/day)', ylabel='Flow-weighted NH4 (mg N/L)', title=weather)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('BSM1 pilot: initial to final policy, 4 updates')
    (a.out/'figures').mkdir(exist_ok=True)
    fig.savefig(a.out/'figures/pilot.png', dpi=160)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
