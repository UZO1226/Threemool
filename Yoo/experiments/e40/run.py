"""e39 고정 예측에서 연속 점수의 판정 경계만 선택하고 평가한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

MODELS = ('base', 'history')
MODES = ('recall80', 'fpr10')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def now():
    return datetime.now(timezone.utc).isoformat()


def choose(score, truth, mode):
    """동점 전체를 함께 판정한다. 같은 TP에서는 FP가 적은 경계를 고른다."""
    score, truth = np.asarray(score).ravel(), np.asarray(truth, dtype=bool).ravel()
    assert score.shape == truth.shape and np.isfinite(score).all()
    assert truth.any() and (~truth).any()
    order = np.argsort(-score, kind='stable')
    s, y = score[order], truth[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.r_[0, np.cumsum(y, dtype=np.int64)[ends]]
    fp = np.r_[0, np.cumsum(~y, dtype=np.int64)[ends]]
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if mode == 'recall80':
        i = int(np.flatnonzero(5 * tp >= 4 * positives)[0])
    elif mode == 'fpr10':
        feasible = np.flatnonzero(10 * fp <= negatives)
        best_tp = tp[feasible].max()
        i = int(feasible[tp[feasible] == best_tp][0])
    else:
        raise ValueError(mode)
    return {'threshold': None if i == 0 else float(s[ends[i - 1]]),
            'rule': 'never_alert' if i == 0 else 'score>=threshold',
            'tp': int(tp[i]), 'fp': int(fp[i]), 'positives': positives, 'negatives': negatives,
            'recall_pct': float(100 * tp[i] / positives),
            'fpr_pct': float(100 * fp[i] / negatives)}


def apply(score, op):
    if op['rule'] == 'never_alert':
        assert op['threshold'] is None
        return np.zeros_like(score, dtype=bool)
    assert op['rule'] == 'score>=threshold' and np.isfinite(op['threshold'])
    return score >= op['threshold']


def metrics(truth, alerts):
    assert truth.shape == alerts.shape and truth.ndim == 3
    tp = (truth & alerts).sum((1, 2)); fp = (~truth & alerts).sum((1, 2))
    pos = truth.sum((1, 2)); neg = (~truth).sum((1, 2))
    total_pos, total_neg = int(pos.sum()), int(neg.sum())
    return {'tp': int(tp.sum()), 'fp': int(fp.sum()),
            'positives': total_pos, 'negatives': total_neg,
            'recall_pct': float(100 * tp.sum() / total_pos) if total_pos else None,
            'fpr_pct': float(100 * fp.sum() / total_neg) if total_neg else None,
            'predicted_count_per_day_member': float(alerts.sum(2).mean()),
            'observed_count_per_day_member': float(truth.sum(2).mean()),
            'day_counts': np.column_stack((tp, fp, pos, neg)).tolist()}


def delta(a, b):
    """b-a, 날짜를 함께 재표집한다. 구성원은 독립 seed로 세지 않는다."""
    aa, bb = np.array(a['day_counts']), np.array(b['day_counts'])
    assert len(aa) == len(bb) == 40 and np.array_equal(aa[:, 2:], bb[:, 2:])
    rng = np.random.default_rng(4000)
    starts = rng.integers(40, size=(2000, 6))
    ids = ((starts[:, :, None] + np.arange(7)) % 40).reshape(2000, -1)[:, :40]
    ar, br = aa[ids].sum(1), bb[ids].sum(1)
    result = {}
    for name, col, denom in (('recall_pct', 0, 2), ('fpr_pct', 1, 3)):
        ok = ar[:, denom] > 0
        d = 100 * (br[ok, col] / br[ok, denom] - ar[ok, col] / ar[ok, denom])
        result[name] = {'difference_pp': b[name] - a[name],
                        'paired_7day_block_95ci_pp': np.quantile(d, [.025, .975]).tolist()}
    return result


def dominates(a, b):
    """a가 b보다 두 점추정치에서 나쁘지 않고 하나 이상 좋은지 확인한다."""
    return bool(a['recall_pct'] >= b['recall_pct'] and a['fpr_pct'] <= b['fpr_pct']
                and (a['recall_pct'] > b['recall_pct'] or a['fpr_pct'] < b['fpr_pct']))


def main(source, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    source = source.resolve()
    old_protocol = json.loads((source / 'protocol.json').read_text())
    files = ('predictions.npz', 'models.pkl', 'calibrators.pkl', 'protocol.json',
             'operating_points.json', 'comparison.json', 'manifest.json')
    inputs = {str(source / name): sha(source / name) for name in files}
    old_manifest = json.loads((source / 'manifest.json').read_text())
    for name, key in (('models.pkl', 'model_sha256'), ('calibrators.pkl', 'calibrators_sha256_before_validation'),
                      ('operating_points.json', 'operating_sha256_before_validation'), ('protocol.json', 'protocol_sha256')):
        assert inputs[str(source / name)] == old_manifest[key]
    protocol = {k: old_protocol[k] for k in ('fit_dates', 'calibration_dates', 'validation_dates')}
    protocol.update(locked_at=now(), source=str(source),
        change='Only alert threshold selection uses continuous predicted NH4 instead of isotonic probability.',
        primary='recall80: highest score threshold attaining calibration recall>=80%, thus minimal FPR among feasible thresholds.',
        secondary='fpr10: maximal calibration recall at FPR<=10%; tie break minimal FP, then highest threshold. No alert is an explicit rule, not a probability-range sentinel.',
        comparison='Within each model compare continuous vs e39 isotonic for each profile; separately compare history vs base with continuous thresholds. Report both profiles, never select using validation.',
        screen='Point-estimate dominance requires recall not lower, FPR not higher, at least one strict. Separately report validation target attainment and paired day uncertainty. No automatic adoption.',
        labels='Actual NH4>4 remains truth; raw reference predicted NH4>4. Threshold is an alert score boundary, never a new water-quality limit.',
        scope='Saved e39 conditional recorded-action replay. Model weights, concentration trajectories, DO, physical costs, reward and airflow unchanged. No new model/RL training or test use.',
        inference='Follow-up motivated by previously observed e39 development validation and calibration diagnostics. Frozen thresholds here do not make reused validation an independent confirmatory test.',
        uncertainty='Circular paired 7-day date block bootstrap, 2000 draws, seed4000. Five model members are dependent, not policy seeds.',
        events='Same e39 recorded windows, all and actual NH4 rise>0.1 after airflow decrease, 15/60/180 minutes. Future growth only defines evaluation subgroup.',
        stop='One score-boundary ablation. If validation targets or tradeoff remain unresolved, do not tune more thresholds on validation; return to NH4 transition response and DO limits.')
    save(out / 'protocol.json', protocol)
    manifest = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'started_at': now(), 'input_sha256': inputs, 'protocol_sha256': sha(out / 'protocol.json'),
                'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    z = np.load(source / 'predictions.npz')
    truth_cal = np.repeat(z['actual_calibration'][:, None, :, 0] > 4, 5, axis=1)
    assert truth_cal.shape == (48, 5, 95)
    operating = {}
    for name in MODELS:
        score = z[name + '_calibration_pred'][:, :, :, 0]
        operating[name] = {mode: choose(score, truth_cal, mode) for mode in MODES}
        for mode in MODES:
            x = metrics(truth_cal, apply(score, operating[name][mode]))
            for key in ('tp', 'fp', 'recall_pct', 'fpr_pct'):
                assert x[key] == operating[name][mode][key]
    save(out / 'operating_points.json', {'locked_at': now(), 'models': operating})
    manifest['operating_sha256_before_validation'] = sha(out / 'operating_points.json')
    save(out / 'manifest.json', manifest)
    print('FROZEN', json.dumps(operating), flush=True)
    # The validation arrays are first accessed only after the score boundaries are saved.
    truth = np.repeat(z['actual'][:, None, :, 0] > 4, 5, axis=1)
    assert truth.shape == (40, 5, 95)
    old = json.loads((source / 'comparison.json').read_text())
    val, differences, screens, arrays, events = {}, {}, {}, {}, []
    for name in MODELS:
        score = z[name + '_pred'][:, :, :, 0]
        val[name] = {'fixed4': metrics(truth, score > 4)}
        differences[name], screens[name] = {}, {}
        for mode in MODES:
            new_alerts = apply(score, operating[name][mode])
            iso_alerts = z[name + '_alert_' + mode]
            a, b = metrics(truth, iso_alerts), metrics(truth, new_alerts)
            for key in ('tp', 'fp', 'recall_pct', 'fpr_pct'):
                assert a[key] == old['validation'][name]['nh4'][mode][key]
            val[name]['isotonic_' + mode] = a
            val[name]['continuous_' + mode] = b
            arrays[name + '_' + mode] = new_alerts
            differences[name][mode] = delta(a, b)
            screens[name][mode] = {'continuous_dominates_isotonic': dominates(b, a),
                'validation_target_met': b['recall_pct'] >= 80 if mode == 'recall80' else b['fpr_pct'] <= 10,
                'nonzero_recall': b['recall_pct'] > 0}
        for step in (1, 4, 12):
            for group in ('all', 'air_decrease_and_nh4_rising'):
                actual = z['event_actual']
                mask = np.ones(len(actual), bool) if group == 'all' else ((z['event_dq'] < 0) & (actual[:, step, 0] - actual[:, 0, 0] > .1))
                tr = np.repeat((actual[mask, step, 0] > 4)[:, None, None], 5, axis=1)
                score_event = z[name + '_events'][mask, :, step, 0][:, :, None]
                for mode in MODES:
                    item = metrics(tr, apply(score_event, operating[name][mode]))
                    item.pop('day_counts')
                    events.append(dict(model=name, mode=mode, group=group, minutes=15 * step, windows=int(mask.sum()), **item))
    across = {mode: delta(val['base']['continuous_' + mode], val['history']['continuous_' + mode]) for mode in MODES}
    across_screen = {mode: dominates(val['history']['continuous_' + mode], val['base']['continuous_' + mode]) for mode in MODES}
    result = {'validation': val, 'continuous_minus_isotonic': differences,
              'history_minus_base_continuous': across, 'within_model_screen': screens,
              'history_dominates_base_continuous': across_screen, 'events': events,
              'physical_outputs_unchanged': True, 'new_model_training': False, 'RL_training': False,
              'test_evaluated': False, 'automatic_adoption': False}
    save(out / 'comparison.json', result)
    np.savez_compressed(out / 'alerts.npz', **arrays)
    assert all(sha(path) == value for path, value in inputs.items())
    assert sha(out / 'operating_points.json') == manifest['operating_sha256_before_validation']
    assert sha(out / 'protocol.json') == manifest['protocol_sha256']
    manifest.update(finished_at=now(), elapsed_seconds=time.monotonic() - start, checks_passed=True)
    save(out / 'manifest.json', manifest)
    print('DONE', manifest['elapsed_seconds'], json.dumps(screens), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.out)
