"""DARROW 기록 공기 변화의 조건부 수질 반응과 모델 재현성을 대조한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from darrow_env import Dynamics, load_days, STATE, EXO
from common import sha, save

H = 12
HORIZONS = (1, 4, 12)
DEADBAND = np.array([.1, .1, .05, .05, .05])
FEATURES = STATE + ['trend_' + s for s in STATE] + EXO + ['previous_air'] + ['previous_fraction_' + str(i) for i in range(1, 6)]


def context(g, t):
    # No current action or post-event outcome enters the matching covariates.
    previous = g['zone'][t-1]
    return np.r_[g['state'][t], (g['state'][t] - g['state'][t-4]) / 4,
                 g['exo'][t], previous.sum(), previous / max(previous.sum(), 1)]


def windows(days, dates):
    return [{'day': d, 't': t, 'dq': float(days[d]['air'][t] - days[d]['air'][t-1]),
             'x': context(days[d], t)} for d in dates for t in range(4, 96-H)]


def events(pool, threshold):
    selected, last = [], {}
    for w in pool:
        if abs(w['dq']) < threshold or w['t'] < last.get(w['day'], -100) + H:
            continue
        selected.append(w)
        last[w['day']] = w['t']
    return selected


def nearest_crossday(query, controls, center, scale):
    q = (np.stack([w['x'] for w in query]) - center) / scale
    c = (np.stack([w['x'] for w in controls]) - center) / scale
    model = NearestNeighbors(n_neighbors=min(200, len(c)), n_jobs=1).fit(c)
    distances, ids = model.kneighbors(q)
    result = []
    for i, w in enumerate(query):
        valid = [j for j in range(ids.shape[1]) if controls[ids[i, j]]['day'] != w['day']]
        if not valid:
            result.append(None)
            continue
        j = valid[0]
        k = int(ids[i, j])
        result.append({'control_index': k, 'rms_iqr_distance': float(distances[i, j] / np.sqrt(q.shape[1])),
                       'max_iqr_difference': float(np.max(np.abs(q[i] - c[k])))})
    return result


def simulate(windows_, days, dyn):
    n, m = len(windows_), dyn.n_members
    initial = np.stack([days[w['day']]['state'][w['t']] for w in windows_])
    state = np.repeat(initial, m, axis=0)
    member = np.tile(np.arange(m), n)
    previous = np.repeat(np.stack([days[w['day']]['zone'][w['t']-1] for w in windows_]), m, axis=0)
    actual = np.stack([days[w['day']]['state'][w['t']:w['t']+H+1] for w in windows_])
    pred = [state.reshape(n, m, 5).copy()]
    for h in range(H):
        exo = np.repeat(np.stack([days[w['day']]['exo'][w['t']+h] for w in windows_]), m, axis=0)
        zone = np.repeat(np.stack([days[w['day']]['zone'][w['t']+h] for w in windows_]), m, axis=0)
        state = dyn.step(state, exo, zone, previous, member)
        assert np.all(np.isfinite(state))
        pred.append(state.reshape(n, m, 5).copy())
        previous = zone
    return actual, np.stack(pred, 2)


def block_ci(values, day_ids, n_days):
    sums = np.bincount(day_ids, weights=values, minlength=n_days)
    counts = np.bincount(day_ids, minlength=n_days)
    rng = np.random.default_rng(3700)
    starts = rng.integers(n_days, size=(2000, int(np.ceil(n_days/7))))
    ids = ((starts[:, :, None] + np.arange(7)) % n_days).reshape(2000, -1)[:, :n_days]
    denom = counts[ids].sum(1)
    good = denom > 0
    boot = sums[ids[good]].sum(1) / denom[good]
    return np.quantile(boot, [.025, .975]).tolist()


def classes(x, tolerance):
    return np.where(x > tolerance, 1, np.where(x < -tolerance, -1, 0))


def summarize(actual, pred, subset, window_list, dates):
    ids = np.flatnonzero(subset)
    if not len(ids):
        return {'events': 0, 'days': 0, 'metrics': []}
    a, p = actual[ids], pred[ids]
    did = np.array([dates.index(window_list[i]['day']) for i in ids])
    rows = []
    for h in HORIZONS:
        delta = a[:, h] - a[:, 0]
        members = p[:, :, h] - p[:, :, 0]
        mean = members.mean(1)
        for j, name in enumerate(STATE):
            ac, pc = classes(delta[:, j], DEADBAND[j]), classes(mean[:, j], DEADBAND[j])
            recalls = [float((pc[ac == k] == k).mean()) for k in (-1, 0, 1) if np.any(ac == k)]
            error = np.abs(mean[:, j] - delta[:, j])
            persistence = np.abs(delta[:, j])
            rows.append({'target': name, 'minutes': h*15,
                         'observed_mean_delta': float(delta[:, j].mean()), 'predicted_mean_delta': float(mean[:, j].mean()),
                         'mae': float(error.mean()), 'persistence_mae': float(persistence.mean()),
                         'mae_minus_persistence': float((error-persistence).mean()),
                         'paired_7day_block_95ci': block_ci(error-persistence, did, len(dates)),
                         'direction_accuracy_pct': float((ac == pc).mean()*100),
                         'direction_balanced_accuracy_pct': float(np.mean(recalls)*100),
                         'observed_direction_counts': {str(k): int((ac == k).sum()) for k in (-1, 0, 1)},
                         'mean_member_mae': float(np.abs(members[:, :, j]-delta[:, None, j]).mean()),
                         'member_mean_mae_by_member': np.abs(members[:, :, j]-delta[:, None, j]).mean(0).tolist()})
    peaks = {}
    for j, name in enumerate(STATE):
        # A trajectory extremum is descriptive; it is not an identified causal delay.
        ap = np.argmax(np.abs(a[:, 1:, j]-a[:, :1, j]), axis=1)+1
        pm = p.mean(1)
        pp = np.argmax(np.abs(pm[:, 1:, j]-pm[:, :1, j]), axis=1)+1
        peaks[name] = {'observed_peak_median_minutes': float(np.median(ap)*15),
                       'predicted_peak_median_minutes': float(np.median(pp)*15),
                       'peak_time_absolute_error_median_minutes': float(np.median(np.abs(ap-pp))*15),
                       'observed_peak_at_3h_pct': float((ap == H).mean()*100)}
    return {'events': len(ids), 'days': len(np.unique(did)), 'metrics': rows, 'trajectory_extrema': peaks}


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    splits, days = load_days(a.prepared / 'data.pkl')
    assert len(splits['train']) == 188 and len(splits['val']) == 40
    train_pool = windows(days, splits['train'])
    dq = np.array([abs(w['dq']) for w in train_pool])
    large = float(max(500., np.quantile(dq, .9)))
    small = float(min(100., np.quantile(dq, .25)))
    train_events = events(train_pool, large)
    train_controls = [w for w in train_pool if abs(w['dq']) <= small]
    x = np.stack([w['x'] for w in train_pool])
    center = np.median(x, axis=0)
    scale = np.maximum(np.quantile(x, .75, axis=0)-np.quantile(x, .25, axis=0), 1e-6)
    train_matches = nearest_crossday(train_events, train_controls, center, scale)
    distance_limit = float(np.quantile([m['rms_iqr_distance'] for m in train_matches if m], .9))
    protocol = {'locked_at': datetime.now(timezone.utc).isoformat(), 'horizons_minutes': [15, 60, 180],
                'large_change_m3_h': large, 'stable_change_m3_h': small,
                'event_rule': 'train abs delta Q q90, floor 500; chronological nonoverlap of 12 transitions within day; t>=4 and t+12<=95',
                'controls_rule': 'abs delta Q <= min(train q25,100); nearest pre-event state/trend/exogenous/previous-zone-share on DIFFERENT day; replacement allowed',
                'features': FEATURES, 'center': center.tolist(), 'iqr_scale': scale.tolist(),
                'rms_distance_limit': distance_limit, 'maximum_single_feature_iqr_distance': 1.,
                'matching_rule': 'nearest other-day stable window; eligible if RMS <= train nearest-distance q90 AND every standardized difference <=1; not causal common-support certification',
                'direction_deadband_mg_L': DEADBAND.tolist(),
                'minimum_descriptive_pair_support': {'pairs': 30, 'event_days': 10},
                'reference': 'persistence state at start, same recorded future actions/exogenous inputs; model recursively predicts state, no future state injection',
                'decision': 'Do not certify action effects from observational matching. If model fails persistence at key NH4/DO horizons or matching is sparse, withhold new RL ranking/adoption. Better predictions are necessary evidence, not a causal safety certificate.',
                'inference': 'exploratory development val; date circular 7day block bootstrap 2000; no p-value claims or multiple-testing success selection',
                'training': False, 'test_evaluated': False}
    save(a.out / 'protocol.json', protocol)
    inputs = {str(p): sha(p) for p in (a.prepared/'data.pkl', a.corrected)}
    manifest = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'script_sha256': sha(Path(__file__)), 'input_sha256': inputs, 'protocol_sha256': sha(a.out/'protocol.json'),
                'train_days': 188, 'validation_days': 40, 'training_performed': False, 'test_evaluated': False}
    save(a.out/'manifest.json', manifest)
    print('LOCKED', large, small, distance_limit, 'train events', len(train_events), flush=True)
    val_pool = windows(days, splits['val'])
    ev = events(val_pool, large)
    controls = [w for w in val_pool if abs(w['dq']) <= small]
    matches = nearest_crossday(ev, controls, center, scale)
    pairs = []
    for i, m in enumerate(matches):
        if m is None:
            continue
        c = controls[m['control_index']]
        accepted = m['rms_iqr_distance'] <= distance_limit and m['max_iqr_difference'] <= 1
        pairs.append({'event_index': i, 'control_index': m['control_index'], 'event_day': ev[i]['day'],
                      'event_t': ev[i]['t'], 'event_dq': ev[i]['dq'], 'control_day': c['day'], 'control_t': c['t'],
                      'control_dq': c['dq'], 'accepted': bool(accepted), **m})
    save(a.out/'matches.json', pairs)
    print('VAL events', len(ev), 'controls', len(controls), 'accepted', sum(p['accepted'] for p in pairs), flush=True)
    dyn = Dynamics(a.corrected)
    all_windows = ev + controls
    actual, pred = simulate(all_windows, days, dyn)
    n = len(ev)
    np.savez_compressed(a.out/'trajectories.npz', actual=actual, pred=pred,
                        days=np.array([w['day'] for w in all_windows]), t=np.array([w['t'] for w in all_windows]),
                        dq=np.array([w['dq'] for w in all_windows]), context=np.stack([w['x'] for w in all_windows]))
    results = {}
    increase = np.array([w['dq'] > 0 for w in ev])
    for name, mask in [('all_changes', np.ones(n, bool)), ('increase', increase), ('decrease', ~increase)]:
        results[name] = summarize(actual[:n], pred[:n], mask, ev, splits['val'])
    accepted = [p for p in pairs if p['accepted']]
    pair_rows = []
    for h in HORIZONS:
        for j, target in enumerate(STATE):
            for group in ('increase', 'decrease'):
                pp = [p for p in accepted if (p['event_dq'] > 0) == (group == 'increase')]
                if not pp:
                    continue
                e = np.array([p['event_index'] for p in pp])
                c = np.array([n+p['control_index'] for p in pp])
                ad = (actual[e, h, j]-actual[e, 0, j])-(actual[c, h, j]-actual[c, 0, j])
                pm = pred.mean(1)
                pdiff = (pm[e, h, j]-pm[e, 0, j])-(pm[c, h, j]-pm[c, 0, j])
                pair_rows.append({'group': group, 'minutes': h*15, 'target': target, 'pairs': len(pp),
                                  'observed_mean_change_minus_control': float(ad.mean()),
                                  'predicted_mean_change_minus_control': float(pdiff.mean()),
                                  'contrast_mae': float(np.abs(ad-pdiff).mean()),
                                  'causal_interpretation': False})
    accepted_days = len(set(p['event_day'] for p in accepted))
    support = {'train_events': len(train_events), 'train_stable_windows': len(train_controls),
               'val_events': n, 'val_event_days': len(set(w['day'] for w in ev)), 'val_stable_windows': len(controls),
               'accepted_pairs': len(accepted), 'accepted_event_days': accepted_days,
               'unique_control_windows': len(set((p['control_day'],p['control_t']) for p in accepted)),
               'minimum_descriptive_support_met': len(accepted) >= 30 and accepted_days >= 10,
               'by_direction': {g: sum((p['event_dq'] > 0) == (g == 'increase') for p in accepted) for g in ('increase','decrease')}}
    error_failure = [r for r in results['all_changes']['metrics'] if r['target'] != 'no3' and r['paired_7day_block_95ci'][0] > 0]
    result = {'support': support, 'response': results, 'matched_contrasts': pair_rows,
              'worse_than_persistence_exploratory_intervals': error_failure,
              'no_automatic_environment_or_policy_adoption': True,
              'decision': 'withhold_control_effect_claim; evaluate evidence by target and horizon, no causal identification'}
    save(a.out/'results.json', result)
    assert sha(a.out/'protocol.json') == manifest['protocol_sha256']
    assert all(sha(Path(p)) == v for p,v in inputs.items())
    manifest.update(elapsed_seconds=time.monotonic()-start, checks_passed=True, finished_at=datetime.now(timezone.utc).isoformat())
    save(a.out/'manifest.json', manifest)
    print('DONE', json.dumps(support), manifest['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
