"""저장된 경계와 분류를 별도 ROC 계산 및 confusion matrix로 검산한다."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    opfile = json.loads((out / 'operating_points.json').read_text())
    op = opfile['models']
    result = json.loads((out / 'comparison.json').read_text())
    source = Path(protocol['source'])
    z = np.load(source / 'predictions.npz')
    saved_alerts = np.load(out / 'alerts.npz')
    fit, cal, val = [protocol[k] for k in ('fit_dates', 'calibration_dates', 'validation_dates')]
    assert (len(fit), len(cal), len(val)) == (140, 48, 40)
    assert max(fit) < min(cal) and max(cal) < min(val)
    assert not (set(fit) & set(cal) or set(cal) & set(val) or set(fit) & set(val))
    assert protocol['locked_at'] <= opfile['locked_at'] <= manifest['finished_at']
    ycal = np.broadcast_to(z['actual_calibration'][:, None, :, 0] > 4, (48, 5, 95))
    yval = np.broadcast_to(z['actual'][:, None, :, 0] > 4, (40, 5, 95))
    metrics_checked = 0
    for name in ('base', 'history'):
        score = z[name + '_calibration_pred'][:, :, :, 0]
        fpr, tpr, boundaries = roc_curve(ycal.ravel(), score.ravel(), drop_intermediate=False)
        for mode in ('recall80', 'fpr10'):
            chosen = op[name][mode]
            if mode == 'recall80':
                index = np.flatnonzero(tpr >= .8)[0]
                assert fpr[index] == fpr[tpr >= .8].min()
            else:
                feasible = np.flatnonzero(fpr <= .1)
                index = feasible[tpr[feasible] == tpr[feasible].max()][0]
            assert np.isclose(100 * tpr[index], chosen['recall_pct'])
            assert np.isclose(100 * fpr[index], chosen['fpr_pct'])
            if chosen['threshold'] is None:
                assert chosen['rule'] == 'never_alert' and index == 0
                ca = np.zeros(score.shape, bool)
                va = np.zeros(yval.shape, bool)
            else:
                assert boundaries[index] == chosen['threshold']
                ca = score >= chosen['threshold']
                va = z[name + '_pred'][:, :, :, 0] >= chosen['threshold']
            assert np.array_equal(va, saved_alerts[name + '_' + mode])
            assert np.count_nonzero(ca & ycal) == chosen['tp']
            assert np.count_nonzero(ca & ~ycal) == chosen['fp']
        for condition, target in result['validation'][name].items():
            if condition == 'fixed4':
                aa = z[name + '_pred'][:, :, :, 0] > 4
            elif condition.startswith('isotonic_'):
                aa = z[name + '_alert_' + condition.removeprefix('isotonic_')]
            else:
                aa = saved_alerts[name + '_' + condition.removeprefix('continuous_')]
            tn, fp, fn, tp = confusion_matrix(yval.ravel(), aa.ravel(), labels=[False, True]).ravel()
            assert (int(tp), int(fp), int(tp + fn), int(tn + fp)) == tuple(target[k] for k in ('tp', 'fp', 'positives', 'negatives'))
            assert np.isclose(target['recall_pct'], 100 * tp / (tp + fn))
            assert np.isclose(target['fpr_pct'], 100 * fp / (fp + tn))
            assert np.isclose(target['predicted_count_per_day_member'], aa.sum() / 200)
            for day, counts in enumerate(target['day_counts']):
                tn1, fp1, fn1, tp1 = confusion_matrix(yval[day].ravel(), aa[day].ravel(), labels=[False, True]).ravel()
                assert counts == [int(tp1), int(fp1), int(tp1 + fn1), int(tn1 + fp1)]
            metrics_checked += 1
    for event in result['events']:
        step = event['minutes'] // 15
        actual = z['event_actual']
        mask = np.ones(len(actual), bool) if event['group'] == 'all' else ((z['event_dq'] < 0) & (actual[:, step, 0] - actual[:, 0, 0] > .1))
        truth = np.broadcast_to(actual[mask, step, 0][:, None] > 4, (int(mask.sum()), 5))
        chosen = op[event['model']][event['mode']]
        alerts = np.zeros(truth.shape, bool) if chosen['threshold'] is None else z[event['model'] + '_events'][mask, :, step, 0] >= chosen['threshold']
        tn, fp, fn, tp = confusion_matrix(truth.ravel(), alerts.ravel(), labels=[False, True]).ravel()
        assert (int(tp), int(fp), int(tp + fn), int(tn + fp)) == tuple(event[k] for k in ('tp', 'fp', 'positives', 'negatives'))
        assert event['windows'] == mask.sum()
        assert np.isclose(event['predicted_count_per_window_member'], alerts.mean())
        metrics_checked += 1
    assert all(sha(path) == value for path, value in manifest['input_sha256'].items())
    assert sha(out / 'protocol.json') == manifest['protocol_sha256']
    assert sha(out / 'operating_points.json') == manifest['operating_sha256_before_validation']
    for filename, digest in manifest['source_sha256'].items():
        assert sha(Path(__file__).parent / filename) == digest
    checks = {'passed': True, 'calibration_operating_points_independently_verified': 4,
              'classification_checks': metrics_checked, 'daily_confusion_checks': 400,
              'source_and_frozen_input_hashes_unchanged': True, 'disjoint_dates': True}
    (out / 'verification.json').write_text(json.dumps(checks, indent=2))
    print(checks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
