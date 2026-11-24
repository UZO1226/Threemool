"""Update a fixed late-air target using only the observed cut-branch prefix."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
for folder in ('e90', 'e97'):
    sys.path.insert(0, str(HERE.parent / folder))
import rebound
import intervals

prior = intervals.prior
S = prior.S


class ObservedStates:
    def __init__(self, dyn):
        self.dyn = dyn
        self.prefix = []

    def step(self, *args):
        state = self.dyn.step(*args)
        if len(self.prefix) < 12:
            self.prefix.append(state.copy())
        return state


def measured(states, volume, start, end):
    state = states[:, end-1]
    return np.column_stack((state[:, 0], state[:, 2:].max(1), volume[:, end-1] * 4,
                            volume[:, start:end].mean(1) * 4))


def replay(rows, archive, keys, q, dyn, agent, expected, training):
    panels, targets = [], []
    for start in range(0, len(keys), 250):
        sl = slice(start, start+250)
        args = (rows, archive, keys[sl], q[sl])
        trace = ObservedStates(dyn)
        b, actions = rebound.simulate(*args, agent, trace, True)
        states = np.stack(trace.prefix, axis=1)
        volume = b[:, :, 0]
        np.testing.assert_allclose(volume, .25 * S.R.action_to_air(actions), atol=1e-8, rtol=0)
        panel = np.column_stack([measured(states, volume, 0, 4), measured(states, volume, 4, 12)])
        for offset, t in ((0,3),(4,11)):
            np.testing.assert_array_equal(panel[:,offset] > 4, b[:,t,1])
            np.testing.assert_array_equal(panel[:,offset+1] > 2, b[:,t,2])
        horizon = b.cumsum(1)[:, [3,11,23], :]
        if training:
            a = S.P.branch(rows, dyn, agent, keys[sl], q[sl], 0., archive)
            np.testing.assert_allclose(a[:,-1], expected[sl,0], atol=1e-7, rtol=0)
            np.testing.assert_allclose(horizon[:,-1], expected[sl,1], atol=1e-7, rtol=0)
        else:
            a = expected[sl,0]
            np.testing.assert_allclose(horizon, expected[sl,1], atol=1e-7, rtol=0)
        targets.append(volume[:,12:].sum(1) - (a[:,2,0]-a[:,1,0]))
        panels.append(panel)
    return np.concatenate(panels), np.concatenate(targets)


def analyze(day, first, selected, y, pred, center):
    w = S.weights(day[first])
    err = np.average((pred[first]-y[first,None])**2, weights=w, axis=0)
    constant = np.average((center-y[first])**2, weights=w)
    def skill(reference):
        return (1-err/reference).tolist()
    sign = y[first] < -prior.TOL
    effect = np.average(pred*selected[:,None], weights=S.weights(day), axis=0)
    actual = np.average(y*selected, weights=S.weights(day))
    return dict(n=len(y), candidates=int(first.sum()), selected=int(selected.sum()), rmse=np.sqrt(err).tolist(),
                constant_rmse=float(np.sqrt(constant)), vs_constant=skill(constant), vs_initial=skill(err[0]),
                vs_previous=skill(err[[0,0,1]]),
                accuracy=np.average((pred[first] < -prior.TOL)==sign[:,None],weights=w,axis=0).tolist(),
                constant_accuracy=float(np.average((center < -prior.TOL)==sign,weights=w)),
                pred_effect=effect.tolist(), actual_effect=float(actual))


def run(out):
    S.torch.set_num_threads(1)
    assert all(S.digest(Path(k)) == v for k,v in prior.ref('e77')['sha256'].items())
    dates, rows = S.E.load_train()
    assert dates[139] < dates[140]
    dyn = S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        agent = S.E.restore(seed)
        root = S.ROOT
        paths = [root/'e88/main'/f'{seed}_early_down.npz',root/'e78/main'/f'{seed}_train48_down.npz',
                 root/'e88/main'/f'probe{seed}.npz']
        assert [S.digest(p) for p in paths] == prior.ref('e89')[seed]['input_sha256']
        early, k = S.read_candidates(paths[0], dates[:140])
        x, q = S.before_action(rows[:140], early, k)
        with np.load(paths[2]) as z:
            np.testing.assert_array_equal(k, z['train_keys'])
            train_panel, train_y = replay(rows[:140], early, k, q, dyn, agent, z['train_totals'], True)
        late, keys = S.read_candidates(paths[1], dates[140:])
        xt, qt = S.before_action(rows[140:], late, keys)
        probe = root/'e85/main'/f'probe{seed}.npz'
        assert S.digest(probe) == prior.ref('e86')[seed]['score_sha256']
        with np.load(probe) as z:
            full, flow = z['keys'], z['q']
            subset = S.headroom.nonoverlap(full)
            np.testing.assert_array_equal(keys, full[subset])
            np.testing.assert_array_equal(qt, flow[subset])
            panel, truth = replay(rows[140:], late, full, flow, dyn, agent, z['totals'], False)
        panel, truth = panel[subset], truth[subset]
        weights = S.weights(k[:,0])
        center = float(np.average(train_y, weights=weights))
        preds = []
        for width in (0,4,8):
            train = np.column_stack((x, train_panel[:,:width]))
            evaluation = np.column_stack((xt, panel[:,:width]))
            scaler = StandardScaler().fit(train, sample_weight=weights)
            reg = Ridge(alpha=1.).fit(scaler.transform(train), train_y, sample_weight=weights)
            preds.append(reg.predict(scaler.transform(evaluation)))
        pred = np.column_stack(preds)
        with np.load(root/'e97/main'/f'interval{seed}.npz') as z:
            np.testing.assert_array_equal(keys,z['keys'])
            np.testing.assert_allclose(truth, z['truth'][:,2], atol=1e-7, rtol=0)
            np.testing.assert_allclose(pred[:,0], z['pred'][:,2], atol=1e-7, rtol=0)
            np.testing.assert_allclose(center, z['center'][2], atol=1e-7, rtol=0)
            first,supported,selected = z['first'],z['supported'],z['selected']
        groups = {}
        for name,mask in [('supported',supported),('all',np.ones(len(keys),bool))]:
            g = analyze(keys[mask,0], first[mask], selected[mask], truth[mask], pred[mask], center)
            old = prior.ref('e97')[seed]['groups'][name]
            assert (g['n'],g['candidates'],g['selected']) == (old['n'],old['candidates'],old['selected'])
            np.testing.assert_allclose([g['rmse'][0],g['actual_effect']], [old['rmse'][2],old['actual_effect'][2]], atol=1e-7,rtol=0)
            groups[name] = g
        records.append(dict(seed=seed, train_n=len(k), groups=groups))
        np.savez_compressed(out/f'landmark{seed}.npz', keys=keys, panel=panel, pred=pred, truth=truth,
                            center=center, first=first, selected=selected, supported=supported)
        print(seed, groups['supported']['vs_initial'], flush=True)
    (out/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    states = np.tile(np.arange(24.)[None,:,None], (1,1,5))
    volume = np.arange(24.)[None,:]
    for start,end in [(0,4),(4,12)]:
        value = measured(states, volume, start, end)
        np.testing.assert_array_equal(value[0],[end-1,end-1,4*(end-1),2*(start+end-1)])
        future, future_air = states.copy(), volume.copy()
        future[:,end:] = future_air[:,end:] = np.nan
        np.testing.assert_array_equal(measured(future,future_air,start,end),value)
    if args.out:
        run(args.out)
