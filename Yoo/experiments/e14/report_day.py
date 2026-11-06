"""하루 학습 대조의 고정 기준 성능과 조건별 실패를 집계한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from day_trial import GROUPS, arrays, pack, group_masks, load_model, routing_frame, sha, write
from trial import score, replay


def metrics(p, t, mask, threshold):
    p, t = p[mask], t[mask]
    if not len(p): return dict(count=0)
    actual, forecast = t > threshold, p > threshold
    tp, fp, fn, tn = [int(v.sum()) for v in (actual & forecast, ~actual & forecast, actual & ~forecast, ~actual & ~forecast)]
    return dict(count=len(p), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp/(tp+fn) if tp+fn else None, fpr=fp/(fp+tn) if fp+tn else None,
                bias=float((p-t).mean()), mae=float(np.abs(p-t).mean()))


def main(a):
    torch.set_num_threads(1)
    report = json.loads((a.trial / 'results.json').read_text())
    assert 'completed_at' in report and len(report['training']) == 8 and len(report['evaluation']) == 84
    for path in (a.raw, a.data): assert sha(path) == report['input_sha256'][str(path)]
    for r in report['training']:
        assert sha(a.trial / f"{r['mode']}_s{r['seed']}.pt") == r['model_sha256']
    data = pd.read_pickle(a.data)
    xs = {s: arrays(dict(data=data['data'], days=report['partition']), s) for s in ('fit', 'inner_holdout')}
    frame = routing_frame(a.raw)
    aux = np.stack([frame.reindex(data['data'].loc[d].index).to_numpy() for d in report['partition']['fit']])
    truth = {s: np.stack((x['nh4'][:, 13:], x['do'][:, 13:]), -1) for s, x in xs.items()}
    masks = {}
    for split, x in xs.items():
        _, exog = pack(x)
        g = group_masks(torch.tensor(exog[:, 12:-1]).float(), torch.tensor(report['temperature_bounds'])).numpy()
        time = np.broadcast_to(np.arange(1, 276), g.shape[:2])
        masks[split] = {name: g[:, :, k] for k, name in enumerate(GROUPS)}
        masks[split].update(steps_1_36=time <= 36, steps_37_144=(time > 36) & (time <= 144), steps_145_275=time > 144)
    diagnostic = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      training_commit=report['commit'], same_holdout_posthoc_sweep=True,
                      deployment_threshold_changed=False, input_sha256=report['input_sha256'], roc=[], strata=[])
    modes = ('reference', 'start_multi', 'short_mse', 'day_mse', 'day_global', 'day_group')
    selected = [r for r in report['evaluation'] if r['split'] == 'inner_holdout' and r['reset_steps'] == 275]
    for r in selected:
        path = a.trial / f"{r['mode']}_s{r['seed']}_holdout.npz"
        assert sha(path) == r['prediction_sha256']
        with np.load(path) as z: pred = z['pred']
        assert np.isfinite(pred).all() and pred.shape == truth['inner_holdout'].shape
        assert all(r[k] == v for k, v in score(pred, xs['inner_holdout']).items())
        predictions = dict(inner_holdout=pred)
        if r['mode'] in ('start_multi', 'day_mse', 'day_global', 'day_group'):
            if r['mode'] == 'start_multi':
                source = Path(next(p for p in report['baseline_sha256'] if p.endswith(f"/multi_mse_s{r['seed']}.pt")))
                assert sha(source) == report['baseline_sha256'][str(source)]
            else: source = a.trial / f"{r['mode']}_s{r['seed']}.pt"
            predictions['fit'] = replay(load_model(source), xs['fit'], aux, 'with_routing', 275)
            original = next(s for s in report['evaluation'] if s['mode'] == r['mode'] and s['seed'] == r['seed'] and s['split'] == 'fit' and s['reset_steps'] == 275)
            assert all(original[k] == v for k, v in score(predictions['fit'], xs['fit']).items())
        for k, name, threshold in ((0, 'nh4', 4), (1, 'do', 2)):
            y, p = (truth['inner_holdout'][:, :, k] > threshold).ravel(), pred[:, :, k].ravel()
            fpr, recall, cutoff = roc_curve(y, p, drop_intermediate=False)
            idx = np.flatnonzero(fpr <= .1)[-1]
            diagnostic['roc'].append(dict(mode=r['mode'], seed=r['seed'], variable=name,
                                           auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                                           fpr=float(fpr[idx]), recall_at_fpr10=float(recall[idx]),
                                           diagnostic_cutoff=float(cutoff[idx]) if np.isfinite(cutoff[idx]) else None))
            for split, prediction in predictions.items():
                for group, mask in masks[split].items():
                    diagnostic['strata'].append(dict(mode=r['mode'], seed=r['seed'], split=split, variable=name, group=group,
                                                      **metrics(prediction[:, :, k], truth[split][:, :, k], mask, threshold)))
    diagnostic['completed_at'] = datetime.now(timezone.utc).isoformat()
    a.out.mkdir(exist_ok=True, parents=True)
    write(a.out / 'trial_summary.json', report)
    write(a.out / 'diagnostics.json', diagnostic)
    md = ['# Yoo/e14 결과', '', f"학습 실행 `{report['commit']}`.",
          f"시작 `{report['started_at']}`, 완료 `{report['completed_at']}` (UTC).", '',
          '내부 holdout 66일, 초기 1시간 뒤 275전이, seed 2개 평균이다. 반복 관찰한 탐색 구간이며 외부 val, test, RL 학습은 사용하지 않았다. 고정된 물리 기준 NH4 > 4 및 DO > 2로 평가했다. 기존 287전이 정책 비용과 섞지 않는다.', '',
          '## 고정 기준 성능', '',
          '| 군 | NH4 recall | NH4 FPR | NH4 MAE | NH4 편향 | 한 스텝 NH4 MAE | DO recall | DO FPR | 비용 MAE |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for mode in modes:
        rs = [r for r in selected if r['mode'] == mode]
        one = [r for r in report['evaluation'] if r['mode'] == mode and r['split'] == 'inner_holdout' and r['reset_steps'] == 1]
        mean = lambda n, k: np.mean([r[n][k] for r in rs])
        md.append(f"| {mode} | {mean('nh4','step_recall')*100:.2f}% | {mean('nh4','fpr')*100:.2f}% | {mean('nh4','mae'):.3f} | {mean('nh4','bias'):+.3f} | {np.mean([r['nh4']['mae'] for r in one]):.3f} | {mean('do','step_recall')*100:.2f}% | {mean('do','fpr')*100:.2f}% | {mean('cost','daily_mae'):.2f} |")
    md += ['', '## seed별 결과', '', '| 군 | seed | NH4 recall | NH4 FPR | NH4 MAE | DO recall | DO FPR |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in selected:
        md.append(f"| {r['mode']} | {r['seed']} | {r['nh4']['step_recall']*100:.2f}% | {r['nh4']['fpr']*100:.2f}% | {r['nh4']['mae']:.3f} | {r['do']['step_recall']*100:.2f}% | {r['do']['fpr']*100:.2f}% |")
    md += ['', '## 사전 gate', '', '| 군 | e12 screening | recall80/FPR10 | tradeoff |', '| --- | --- | --- | --- |']
    for g in report['gates']:
        trade = next((v['passed'] for v in report['tradeoff_gates'] if v['candidate'] == g['candidate']), '대조군')
        md.append(f"| {g['candidate']} | {g['screen_passed']} | {g['absolute_recall80_fpr10']} | {trade} |")
    primary = next(g for g in report['gates'] if g['candidate'] == 'day_group')
    md += ['', '주 후보의 screening 실패 항목:', '']
    for r in primary['per_seed']:
        md.append(f"- seed {r['seed']}, {r['base']}: " + ', '.join(k for k, v in r.items() if v is False))
    md += ['', '## 조건별 하루 FPR', '', '| 군 | 조건 | fit NH4 FPR | holdout NH4 FPR | fit DO FPR | holdout DO FPR |', '| --- | --- | --- | --- | --- | --- |']
    for mode in ('start_multi', 'day_mse', 'day_global', 'day_group'):
        for group in GROUPS:
            values = []
            for variable in ('nh4', 'do'):
                for split in ('fit', 'inner_holdout'):
                    rs = [r for r in diagnostic['strata'] if r['mode'] == mode and r['variable'] == variable and r['split'] == split and r['group'] == group]
                    values.append(f"{np.mean([r['fpr'] for r in rs])*100:.2f}%" if rs[0]['count'] and rs[0]['fpr'] is not None else '해당 없음')
            md.append('| ' + mode + ' | ' + group + ' | ' + ' | '.join(values) + ' |')
    md += ['', '## 주 후보 승수와 유효 갱신 수', '', '| seed | 조건 | NH4 lambda | DO lambda | NH4 유효 갱신 | DO 유효 갱신 |', '| --- | --- | --- | --- | --- | --- |']
    for t in report['training']:
        if t['mode'] != 'day_group': continue
        for k, group in enumerate(GROUPS):
            m, n = t['last']['multiplier'][k], t['supported_updates'][k]
            md.append(f"| {t['seed']} | {group} | {m[0]:.3f} | {m[1]:.3f} | {n[0]} | {n[1]} |")
    md += ['', '## 같은 FPR에서의 구분 능력 진단', '',
           '동일 holdout에서 판정 기준을 사후 이동한 진단이며 환경 기준을 변경하지 않았다. 독립 검증이나 배포 성능이 아니다.', '',
           '| 군 | NH4 ROC AUC | FPR 10% 이하에서 NH4 recall |', '| --- | --- | --- |']
    for mode in modes:
        rs = [r for r in diagnostic['roc'] if r['mode'] == mode and r['variable'] == 'nh4']
        md.append(f"| {mode} | {np.mean([r['auc'] for r in rs]):.3f} | {np.mean([r['recall_at_fpr10'] for r in rs])*100:.2f}% |")
    md += ['', '## 예측 경과 시간별 NH4 편향', '', '| 군 | 1~36스텝 | 37~144스텝 | 145~275스텝 |', '| --- | --- | --- | --- |']
    for mode in modes:
        values = [np.mean([r['bias'] for r in diagnostic['strata'] if r['mode'] == mode and r['variable'] == 'nh4' and r['split'] == 'inner_holdout' and r['group'] == group]) for group in ('steps_1_36', 'steps_37_144', 'steps_145_275')]
        md.append('| ' + mode + ' | ' + ' | '.join(f'{v:+.3f}' for v in values) + ' |')
    md += ['', '![고정 기준에서 탐지와 거짓경보](figures/tradeoff.png)', '',
           '그림의 선은 같은 군의 seed 0과 1을 연결한다. 임계값을 움직인 ROC 곡선이 아니다. 집단별 TP/FP/FN/TN과 표본 수, bias, MAE는 diagnostics.json에 보존했다.', '']
    (a.out / 'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in modes:
        rs = [r for r in selected if r['mode'] == mode]
        ax.plot([r['nh4']['fpr']*100 for r in rs], [r['nh4']['step_recall']*100 for r in rs], 'o-', label=mode)
    ax.axvline(10, color='gray', linestyle=':', linewidth=1)
    ax.set(xlabel='NH4 false positive rate (%)', ylabel='NH4 recall (%)', title='Fixed threshold, 275-step inner holdout; two seeds per line')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (a.out / 'figures').mkdir(exist_ok=True)
    fig.savefig(a.out / 'figures/tradeoff.png', dpi=160)
    plt.close(fig)
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('trial', 'data', 'raw', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
