"""사전 고정한 비교 규칙으로 정책 효율과 환경 민감도를 요약한다."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def crossings(policy, baseline):
    if policy['do_cost'] > 6:
        return []
    found = []
    for lo, hi in zip(baseline[:-1], baseline[1:]):
        if max(lo['do_cost'], hi['do_cost']) > 6:
            continue
        n1, n2, target = lo['nh4_cost'], hi['nh4_cost'], policy['nh4_cost']
        if n1 == n2 or not min(n1, n2) <= target <= max(n1, n2):
            continue
        w = (target - n1) / (n2 - n1)
        volume = (1 - w) * lo['air_m3'] + w * hi['air_m3']
        severity = (1 - w) * lo['nh4_excess_integral_mg_h_L'] + w * hi['nh4_excess_integral_mg_h_L']
        found.append({'offset_pair': [lo['offset_m3_h'], hi['offset_m3_h']],
                      'weight_high_offset': w, 'baseline_air_m3': volume,
                      'policy_air_change_pct': 100 * (policy['air_m3'] / volume - 1),
                      'baseline_nh4_excess_mg_h_L': severity,
                      'policy_excess_minus_baseline': policy['nh4_excess_integral_mg_h_L'] - severity})
    return found


def day_frame(path):
    x = pd.DataFrame(json.loads(path.read_text()))
    assert len(x) == 200 and x.groupby('day').size().eq(5).all()
    return x.groupby('day').mean(numeric_only=True).sort_index()


def paired(a, b):
    assert a.index.equals(b.index) and len(a) == 40
    rng = np.random.default_rng(3600)
    starts = rng.integers(40, size=(2000, 6))
    ids = ((starts[:, :, None] + np.arange(7)) % 40).reshape(2000, -1)[:, :40]
    result = {}
    for k in ('air_m3', 'nh4_cost', 'do_cost', 'nh4_excess_mg_h_L'):
        if k not in a or k not in b:
            continue
        av, bv = a[k].to_numpy(), b[k].to_numpy()
        values = (av[ids] - bv[ids]).mean(1)
        result[k] = {'mean_difference': float((av - bv).mean()),
                     'paired_7day_block_95ci': np.quantile(values, [.025, .975]).tolist()}
        if k == 'air_m3':
            ratios = 100 * (av[ids].mean(1) / bv[ids].mean(1) - 1)
            result['air_change_pct'] = {'mean': float(100 * (av.mean() / bv.mean() - 1)),
                                       'paired_7day_block_95ci': np.quantile(ratios, [.025, .975]).tolist()}
    return result


def main(out):
    rows = json.loads((out / 'rows.json').read_text())
    selection = json.loads((out / 'selection.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['all_checks_passed']
    assert hashlib.sha256((out / 'selection.json').read_bytes()).hexdigest() == manifest['selection_sha256_before_validation']
    val = [r for r in rows if r['split'] == 'val']
    analyses, effects, sensitivity = [], [], []
    for env in ('legacy', 'corrected'):
        base = sorted([r for r in val if r['environment'] == env and r['kind'] == 'bc_offset'], key=lambda r: r['offset_m3_h'])
        assert len(base) == 9
        selected = day_frame(out / f"val_{env}_{selection['name']}_episodes.json")
        for p in [r for r in val if r['environment'] == env and r['kind'] == 'policy']:
            dominating = [b for b in base if b['nh4_cost'] <= p['nh4_cost'] and b['do_cost'] <= 6 and b['air_m3'] < p['air_m3']]
            full = [b for b in dominating if b['nh4_excess_integral_mg_h_L'] <= p['nh4_excess_integral_mg_h_L']]
            analysis = {'environment': env, 'policy': p['name'], 'both_mean_limits': p['both_mean_limits'],
                        'same_nh4_crossings': crossings(p, base),
                        'grid_dominating_offsets': [b['offset_m3_h'] for b in dominating],
                        'grid_dominating_including_severity': [b['offset_m3_h'] for b in full],
                        'vs_train_selected_baseline': paired(day_frame(out / f"val_{env}_{p['name']}_episodes.json"), selected)}
            analyses.append(analysis)
        for seed in (0, 1):
            actor = day_frame(out / f'val_{env}_fcsrl_actor_s{seed}_episodes.json')
            plain = day_frame(out / f'val_{env}_fcsrl_s{seed}_episodes.json')
            effects.append({'environment': env, 'seed': seed, 'actor_minus_fcsrl': paired(actor, plain)})
    for p in [r for r in val if r['environment'] == 'corrected' and r['kind'] == 'policy']:
        old = next(r for r in val if r['environment'] == 'legacy' and r['name'] == p['name'])
        sensitivity.append({'policy': p['name'], 'old_feasible': old['both_mean_limits'], 'new_feasible': p['both_mean_limits'],
                            'corrected_minus_legacy': paired(day_frame(out / f"val_corrected_{p['name']}_episodes.json"),
                                                           day_frame(out / f"val_legacy_{p['name']}_episodes.json"))})
    output = {'selection': selection, 'policy_comparisons': analyses, 'actor_vs_fcsrl': effects,
              'environment_sensitivity': sensitivity, 'evaluation_commit': manifest['code_commit'],
              'report_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note': 'All matches exploratory. No test evaluation, new training, physical adoption, or seed-level significance claim.'}
    save(out / 'comparison.json', output)
    md = ['# e36 기존 정책 효율과 환경 민감도', '',
          f"평가 실행 커밋: `{manifest['code_commit']}`. 학습/시험 분할 평가 없음. 소요 {manifest['elapsed_seconds']:.1f}초.",
          '검증40일 x 모델5구성원, 정책별seed2개. 공기는m³/23.75h, 비용은95개 중 초과 표본 수다.',
          '같은 NH4 비용은 같은 전체 수질을 뜻하지 않는다. DO 조건과 NH4 초과 강도도 함께 본다.', '',
          f"훈련에서 고정한 기준: BC {selection['offset_m3_h']:+.0f}m³/h. 훈련 평균 한도 충족: {selection['train_feasible']}.", '',
          '| 환경 | 정책 | 공기량 | NH4 비용 | DO 비용 | NH4 초과 적분 mg h/L | 평균6/6 충족 |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for r in val:
        md.append(f"| {r['environment']} | {r['name']} | {r['air_m3']:.1f} | {r['nh4_cost']:.3f} | {r['do_cost']:.3f} | {r['nh4_excess_integral_mg_h_L']:.3f} | {r['both_mean_limits']} |")
    md += ['', '## 같은 NH4 비용의 기준 곡선과 비교', '',
           '음수이면 정책이 기준 곡선보다 공기를 적게 사용한다. DO 조건을 충족하는 인접점 사이만 보간한다.',
           '보간은 기술적 곡선 비교다. 해당 보간 제어기의 실제 실행 성능이나 인과 효과가 아니다.', '',
           '| 환경 | 정책 | 공기 변화 범위 % | NH4 비용, DO 조건, 공기량에서 우세한 기준 offset | 초과 강도까지 우세한 offset |',
           '| --- | --- | --- | --- | --- |']
    for x in analyses:
        c = [r['policy_air_change_pct'] for r in x['same_nh4_crossings']]
        desc = f'{min(c):+.2f} ~ {max(c):+.2f}' if c else '비교 불가'
        md.append(f"| {x['environment']} | {x['policy']} | {desc} | {x['grid_dominating_offsets']} | {x['grid_dominating_including_severity']} |")
    md += ['', '## 학습에서 고른 고정 기준과의 날짜 대응 비교', '',
           '수질을 같게 맞춘 비교가 아니다. 날짜별 구성원 평균에 7일 순환 이동블록 bootstrap2000회를 적용했다.',
           '아래 구간은 개발 날짜의 탐색적 불확실성이며 seed 모집단의 신뢰구간이나 다중검정 보정 결과가 아니다.', '',
           '| 환경 | 정책 | 공기 변화 % [95%구간] | NH4 비용 차이 [95%구간] |', '| --- | --- | --- | --- |']
    for x in analyses:
        diff = x['vs_train_selected_baseline']
        a, n = diff['air_change_pct'], diff['nh4_cost']
        md.append(f"| {x['environment']} | {x['policy']} | {a['mean']:+.2f} {a['paired_7day_block_95ci']} | {n['mean_difference']:+.3f} {n['paired_7day_block_95ci']} |")
    md += ['', '## 해석 범위', '',
           '잘못된 기존 환경은 민감도 확인용이다. 수정 환경 역시 기록 수질 재생 오차가 커 현장 평가기로 검증되지 않았다.',
           '실제 관측 수질과 모델 기록 행동 재생은 별도 행이다. 모든 정책은 동일한 일별 reset, 기록 외생 입력/구간 배분을 쓴다.',
           '정책 체크포인트와 준비 파일 SHA256 불변, BC 기존 평가 및 e35 기록 행동 재생 일치를 검산했다.',
           '개별 날짜/구성원 수치는 외부 결과 폴더의 episodes.json들, 나쁜 날짜는 rows.json에 있다.', '']
    (out / 'results.md').write_text('\n'.join(md))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {'ppo': 'tab:blue', 'fcsrl': 'tab:orange', 'fcsrl_actor': 'tab:green'}
    for ax, env in zip(axes, ('legacy', 'corrected')):
        b = sorted([r for r in val if r['environment'] == env and r['kind'] == 'bc_offset'], key=lambda r: r['offset_m3_h'])
        ax.plot([r['nh4_cost'] for r in b], [r['air_m3']/1000 for r in b], 'o-', color='gray', label='BC + fixed airflow')
        for method, color in colors.items():
            ps = [r for r in val if r['environment'] == env and r.get('method') == method and r['kind'] == 'policy']
            ax.scatter([r['nh4_cost'] for r in ps], [r['air_m3']/1000 for r in ps], color=color, s=55, label=method)
            for r in ps:
                ax.annotate(f"s{r['seed']}", (r['nh4_cost'], r['air_m3']/1000), xytext=(4,4), textcoords='offset points', fontsize=8)
        ax.axvline(6, linestyle=':', color='red', label='NH4 mean limit')
        ax.set(title=env, xlabel='Mean NH4 exceedance count / 95')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Air volume (1000 m3 / 23.75 h)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed policy screening in surrogate environments (DO constraint checked separately)')
    fig.tight_layout()
    fig.savefig(out / 'air_quality.png', dpi=160)
    print('REPORT', out / 'results.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
