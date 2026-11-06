"""고정 기준 결과와 조건별 편향, 판정 기준을 움직인 진단을 기록한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from experiment import arrays, score, sha, write


def rates(pred, truth, mask, threshold):
    p, t = pred[mask], truth[mask]
    if len(p) == 0:
        return dict(count=0)
    actual, forecast = t > threshold, p > threshold
    tp, fp, fn, tn = [int(v.sum()) for v in (actual & forecast, ~actual & forecast, actual & ~forecast, ~actual & ~forecast)]
    return dict(count=len(p), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp / (tp+fn) if tp+fn else None, fpr=fp / (fp+tn) if fp+tn else None,
                bias=float((p-t).mean()), mae=float(np.abs(p-t).mean()))


def main(a):
    report = json.loads((a.trial / 'results.json').read_text())
    assert 'completed_at' in report and len(report['training']) == 8
    assert sha(a.data) == report['input_sha256'][str(a.data)]
    for r in report['training']:
        assert sha(a.trial / f"{r['mode']}_s{r['seed']}.pt") == r['model_sha256']
    data = pd.read_pickle(a.data)
    x = arrays(dict(data=data['data'], days=report['partition']), 'inner_holdout')
    fit = arrays(dict(data=data['data'], days=report['partition']), 'fit')
    truth = np.stack((x['nh4'][:, 13:], x['do'][:, 13:]), -1)
    temperature = x['temp'][:, 12:-1]
    bounds = np.quantile(fit['temp'][:, 12:-1], [1/3, 2/3])
    time = np.broadcast_to(np.arange(1, 276), temperature.shape)
    groups = {'all': np.ones(temperature.shape, dtype=bool),
              'temp_lower': temperature < bounds[0],
              'temp_middle': (temperature >= bounds[0]) & (temperature < bounds[1]),
              'temp_upper': temperature >= bounds[1],
              'steps_1_36': time <= 36, 'steps_37_144': (time > 36) & (time <= 144),
              'steps_145_275': time > 144}
    groups.update({f'phase_{k}': x['phase'][:, 12:-1] == k for k in (0, 1, 2)})
    diagnostic = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      training_commit=report['commit'], generated_at=datetime.now(timezone.utc).isoformat(),
                      temperature_bounds_from_fit=bounds.tolist(),
                      same_holdout_posthoc_sweep=True, deployment_threshold_changed=False,
                      no_posthoc_calibration_applied=True, subgroup_analysis_is_descriptive=True,
                      input_sha256=report['input_sha256'], roc=[], strata=[])
    modes = ('reference', 'start_multi', 'multi', 'multi_anchor', 'multi_rate', 'multi_anchor_rate')
    selected = [r for r in report['evaluation'] if r['split'] == 'inner_holdout' and r['reset_steps'] == 275]
    for r in selected:
        path = a.trial / f"{r['mode']}_s{r['seed']}_holdout.npz"
        assert sha(path) == r['prediction_sha256']
        with np.load(path) as archive:
            pred = archive['pred']
        assert pred.shape == truth.shape and np.isfinite(pred).all()
        recomputed = score(pred, x)
        assert all(r[k] == v for k, v in recomputed.items())
        for k, name, threshold in ((0, 'nh4', 4), (1, 'do', 2)):
            y, p = (truth[:, :, k] > threshold).ravel(), pred[:, :, k].ravel()
            fpr, recall, cut = roc_curve(y, p, drop_intermediate=False)
            idx = np.flatnonzero(fpr <= .1)[-1]
            diagnostic['roc'].append(dict(mode=r['mode'], seed=r['seed'], variable=name,
                                           auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                                           fpr=float(fpr[idx]), recall_at_fpr10=float(recall[idx]),
                                           diagnostic_cutoff=float(cut[idx])))
            for group, mask in groups.items():
                diagnostic['strata'].append(dict(mode=r['mode'], seed=r['seed'], variable=name, group=group,
                                                  **rates(pred[:, :, k], truth[:, :, k], mask, threshold)))
    conditions = json.loads(a.conditions.read_text()) if a.conditions else None
    if conditions is not None:
        assert conditions['training_commit'] == report['commit']
        assert conditions['input_sha256'] == report['input_sha256']
        diagnostic['conditions_sha256'] = sha(a.conditions)
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / 'trial_summary.json', report)
    write(a.out / 'diagnostics.json', diagnostic)
    if conditions is not None:
        write(a.out / 'condition_summary.json', conditions)
    md = ['# Yoo/e13 결과', '', f"학습 실행 커밋 `{report['commit']}`.",
          f"학습 시작 `{report['started_at']}`, 완료 `{report['completed_at']}` (UTC).", '',
          '263일 fit, 반복 관찰한 내부 holdout 66일, seed 2개. 외부 val과 test, RL 학습은 사용하지 않았다. 고정된 NH4 > 4, DO > 2 기준으로 초기 1시간 이후 275전이를 평가한다. 비용은 두 위반 스텝 수의 합이며 기존 정책 한도 15와 비교하지 않는다.', '',
          '## 고정 기준에서 seed 평균', '',
          '| 군 | NH4 recall | NH4 FPR | NH4 편향 | NH4 MAE | 한 스텝 NH4 MAE | DO recall | DO FPR | 비용 MAE |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for mode in modes:
        rows = [r for r in selected if r['mode'] == mode]
        one = [r for r in report['evaluation'] if r['mode'] == mode and r['split'] == 'inner_holdout' and r['reset_steps'] == 1]
        mean = lambda name, key: np.mean([r[name][key] for r in rows])
        md.append(f"| {mode} | {mean('nh4','step_recall')*100:.2f}% | {mean('nh4','fpr')*100:.2f}% | {mean('nh4','bias'):+.3f} | {mean('nh4','mae'):.3f} | {np.mean([r['nh4']['mae'] for r in one]):.3f} | {mean('do','step_recall')*100:.2f}% | {mean('do','fpr')*100:.2f}% | {mean('cost','daily_mae'):.2f} |")
    md += ['', '## seed별 고정 기준', '', '| 군 | seed | NH4 recall | NH4 FPR | 한 스텝 NH4 MAE |', '| --- | --- | --- | --- | --- |']
    for r in selected:
        one = next(s for s in report['evaluation'] if s['mode'] == r['mode'] and s['seed'] == r['seed'] and s['split'] == 'inner_holdout' and s['reset_steps'] == 1)
        md.append(f"| {r['mode']} | {r['seed']} | {r['nh4']['step_recall']*100:.2f}% | {r['nh4']['fpr']*100:.2f}% | {one['nh4']['mae']:.3f} |")
    md += ['', '## 사전 gate', '', '| 군 | e12 screening | recall80/FPR10 | tradeoff 개선 |', '| --- | --- | --- | --- |']
    for g in report['gates']:
        trade = next((t['passed'] for t in report['tradeoff_gates'] if t['candidate'] == g['candidate']), '대조군')
        md.append(f"| {g['candidate']} | {g['screen_passed']} | {g['absolute_recall80_fpr10']} | {trade} |")
    md += ['', '주 후보 multi_anchor_rate의 screening 실패 항목:', '']
    g = next(g for g in report['gates'] if g['candidate'] == 'multi_anchor_rate')
    for row in g['per_seed']:
        md.append(f"- seed {row['seed']}, {row['base']}: " + ', '.join(k for k, v in row.items() if v is False))
    md += ['', '## 같은 FPR에서 비교한 구분 능력', '',
           '같은 holdout에서 판정 기준을 사후에 움직인 ROC 진단이다. 해당 농도 기준을 환경에 적용하거나 보정을 수행하지 않았으며 독립 검증 성능으로 보고하지 않는다.', '',
           '| 군 | NH4 ROC AUC | FPR 10% 이하에서 NH4 recall |', '| --- | --- | --- |']
    for mode in modes:
        rows = [r for r in diagnostic['roc'] if r['mode'] == mode and r['variable'] == 'nh4']
        md.append(f"| {mode} | {np.mean([r['auc'] for r in rows]):.3f} | {np.mean([r['recall_at_fpr10'] for r in rows])*100:.2f}% |")
    md += ['', '## 예측 경과 시간에 따른 NH4 편향', '', '| 군 | 1~36스텝 | 37~144스텝 | 145~275스텝 |', '| --- | --- | --- | --- |']
    for mode in modes:
        means = [np.mean([r['bias'] for r in diagnostic['strata'] if r['mode'] == mode and r['variable'] == 'nh4' and r['group'] == group]) for group in ('steps_1_36', 'steps_37_144', 'steps_145_275')]
        md.append('| ' + mode + ' | ' + ' | '.join(f'{v:+.3f}' for v in means) + ' |')
    md += ['', '## 학습 창과 하루 재생의 차이', '', '| 군 | seed | 마지막 창 NH4 FPR | fit 36 재시작 FPR | fit 275 연속 FPR | holdout 275 연속 FPR | 최종 NH4 lambda |', '| --- | --- | --- | --- | --- | --- | --- |']
    for t in report['training']:
        if 'rate' not in t['mode']: continue
        def get(split, reset):
            return next(r for r in report['evaluation'] if r['mode'] == t['mode'] and r['seed'] == t['seed'] and r['split'] == split and r['reset_steps'] == reset)['nh4']['fpr']
        md.append(f"| {t['mode']} | {t['seed']} | {t['last']['window_fpr'][0]*100:.2f}% | {get('fit',36)*100:.2f}% | {get('fit',275)*100:.2f}% | {get('inner_holdout',275)*100:.2f}% | {t['last']['multiplier'][0]:.3f} |")
    if conditions is not None:
        md += ['', '## 평균에 가려진 조건별 거짓경보', '',
               '주 후보 multi_anchor_rate의 seed 평균이다. 수온 구간은 fit의 3분위 경계로 고정했다. 표본 수는 한 seed의 평가 전이 수이고 FPR의 분모는 그중 실제 비위반 수다.', '',
               '| 조건 | fit 전이 수 | fit NH4 FPR | holdout 전이 수 | holdout NH4 FPR |', '| --- | --- | --- | --- | --- |']
        for group in ('all', 'temp_0', 'temp_1', 'temp_2', 'phase_0', 'phase_1', 'phase_2'):
            parts = []
            for split in ('fit', 'inner_holdout'):
                rs = [r for r in conditions['rows'] if r['mode'] == 'multi_anchor_rate' and r['group'] == group and r['split'] == split]
                parts += [str(rs[0]['count']), f"{np.mean([r['fpr'] for r in rs])*100:.2f}%" if rs[0]['count'] else '해당 없음']
            md.append('| ' + group + ' | ' + ' | '.join(parts) + ' |')
        md += ['', '온도 3구간 x 운전 단계 3구간에서 fit FPR을 holdout 비위반 구성 비율로 재가중했다. 실제 모델을 보정하거나 재학습하지 않았다. 구성 비율 변화와 같은 구간 안의 차이를 나눈 기술적 분해이며 인과 분해가 아니다.', '',
               '| 군 | 원래 fit FPR | 재가중 fit FPR | holdout FPR |', '| --- | --- | --- | --- |']
        for mode in ('start_multi', 'multi', 'multi_anchor_rate'):
            rs = [r for r in conditions['decomposition'] if r['mode'] == mode]
            md.append('| ' + mode + ' | ' + ' | '.join(f"{np.mean([r[key] for r in rs])*100:.2f}%" for key in ('fit_fpr', 'reweighted_fit_fpr', 'holdout_fpr')) + ' |')
    md += ['', '온도 및 운전 단계별 표본 수, TP/FP/FN/TN, bias, MAE는 diagnostics.json 및 condition_summary.json에 있다. 조건별 비교는 실측 운전의 관찰적 진단이며 실제 공정 인과 효과가 아니다.', '', '![고정 기준의 탐지와 거짓경보](figures/tradeoff.png)', '']
    (a.out / 'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in modes:
        rows = [r for r in selected if r['mode'] == mode]
        ax.plot([r['nh4']['fpr']*100 for r in rows], [r['nh4']['step_recall']*100 for r in rows], 'o-', label=mode)
        for r in rows:
            ax.annotate(str(r['seed']), (r['nh4']['fpr']*100, r['nh4']['step_recall']*100), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.axvline(10, color='gray', linestyle=':', linewidth=1)
    ax.set(xlabel='NH4 false positive rate (%)', ylabel='NH4 recall (%)', title='Fixed physical threshold, 275-step inner holdout')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (a.out / 'figures').mkdir(exist_ok=True)
    fig.savefig(a.out / 'figures/tradeoff.png', dpi=160)
    plt.close(fig)
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('trial', 'data', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--conditions', type=Path)
    main(parser.parse_args())
