"""Replay observed inputs with one bounded correction to e50's NH4 output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e104'))
import replay as base

E = base.E
GAINS = (0.0, 0.05, 0.10, 0.20)
METRICS = ('nh4_15m', 'nh4_1h', 'nh4_3h', 'segment_mae', 'do_3h',
           'observed_segment', 'predicted_segment')


def memory(air):
    signal = np.zeros(len(air))
    for i, dq in enumerate(np.diff(air), start=1):
        signal[i] = max(2**(-0.25) * signal[i-1], np.clip(dq / 500, 0, 1))
    return signal


def correct(state, drive, gain):
    state[:, 0] = np.maximum(0, state[:, 0] - gain * drive)
    return state


def forecasts(dyn, initial, exo, zone, drive, gain):
    n, m = len(initial), dyn.n_members
    repeat = lambda x: np.repeat(x, m, axis=0)
    member = np.tile(np.arange(m), n)
    state = repeat(initial)
    saved = []
    for t in range(12):
        state = dyn.step(state, repeat(exo[:, t]), repeat(zone[:, t+1]),
                         repeat(zone[:, t]), member)
        state = correct(state, repeat(drive[:, t]), gain)
        if t+1 in base.STEPS:
            saved.append(state.reshape(n, m, 5).copy())
    return np.stack(saved, axis=1)


def metrics(truth, pred):
    nh = pred[..., 0].mean(axis=2)
    observed = truth[:, 12, 0] - truth[:, 4, 0]
    segment = nh[:, 2] - nh[:, 1]
    do = pred[:, 2, :, 2:].max(axis=-1).mean(axis=1)
    return np.column_stack((abs(nh-truth[:, base.STEPS, 0]), abs(segment-observed),
                            abs(do-truth[:, 12, 2:].max(axis=1)), observed, segment))


def cohort(dates, days):
    events = base.W.events(base.W.windows(days, dates), 500.0)
    date_index = {date: i for i, date in enumerate(dates)}
    history = {d: memory(days[d]['zone'].sum(axis=1)) for d in dates}
    ids = np.array([date_index[w['day']] for w in events])
    t = np.array([w['t'] for w in events])
    dq = np.array([w['dq'] for w in events])
    drive = np.stack([history[w['day']][w['t']:w['t']+12] for w in events])
    truth, exo, zone = base.pack(events, days)
    np.testing.assert_allclose(zone[:, 1].sum(1)-zone[:, 0].sum(1), dq, atol=1e-8)
    return ids, t, dq, drive, truth, exo, zone


def evaluate(dyn, dates, days, gains, out, name):
    ids, t, dq, drive, truth, exo, zone = cohort(dates, days)
    preds = np.stack([forecasts(dyn, truth[:, 0], exo, zone, drive, g) for g in gains])
    assert np.isfinite(preds).all()
    np.savez_compressed(out / f'{name}.npz', ids=ids, t=t, dq=dq, truth=truth, preds=preds,
                        dates=np.array(dates, dtype=str), gains=gains)
    if name == 'winter':
        with np.load(E.DATA.parents[1] / 'e104/main/replay.npz') as old:
            for new, key in [(ids, 'date_index'), (t, 't'), (dq, 'dq'), (truth, 'truth')]:
                np.testing.assert_array_equal(new, old[key])
            np.testing.assert_allclose(preds[0], old['rec'], rtol=0, atol=1e-12)
    return ids, dq, np.stack([metrics(truth, p) for p in preds])


def daily_mean(values, ids):
    return np.stack([values[ids == d].mean(axis=0) for d in np.unique(ids)]).mean(axis=0)


def smoke():
    air = np.array([10, 510, 510, 510, 510, 510, 80, 1000])
    response = memory(air)
    np.testing.assert_allclose(response[[0, 1, 5, 7]], [0, 1, .5, 1])
    np.testing.assert_array_equal(memory(air[:5]), response[:5])
    x = np.arange(10, dtype=float).reshape(2, 5)
    y = correct(x.copy(), np.ones(2), 0.2)
    np.testing.assert_array_equal(y[:, 1:], x[:, 1:])
    np.testing.assert_allclose(y[:, 0], [0, 4.8])
    np.testing.assert_array_equal(correct(x.copy(), np.ones(2), 0), x)
    base.smoke()


def run(out):
    ref = json.loads((HERE.parent / 'e77/summary.json').read_text())
    for p in (E.DATA / 'surrogate.pkl', E.DATA / 'data.pkl'):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == ref['sha256'][str(p)]
    dates, rows = E.load_train()
    days = dict(zip(dates, rows))
    dyn = E.Dynamics(E.DATA / 'surrogate.pkl')
    out.mkdir(parents=True, exist_ok=False)
    ids, dq, scores = evaluate(dyn, dates[:140], days, GAINS, out, 'fit')
    up = dq > 0
    loss = [daily_mean(s[up], ids[up])[METRICS.index('segment_mae')] for s in scores]
    best = min(range(1, len(GAINS)), key=lambda i: (loss[i], GAINS[i]))
    gain = GAINS[best]
    lock = dict(gain=gain, candidates=GAINS, fit_increase_segment_mae=loss)
    (out / 'selection.json').write_text(json.dumps(lock, indent=2)+'\n')
    print('Locked selection', lock, flush=True)
    evaluate(dyn, dates[140:], days, (0.0, gain), out, 'confirm')
    split, days = E.S.base.load_selected(E.DATA / 'data.pkl')
    assert list(split['train']) == dates and len(split['val']) == 40
    evaluate(dyn, split['val'], days, (0.0, gain), out, 'winter')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    smoke()
    if args.out:
        run(args.out)
