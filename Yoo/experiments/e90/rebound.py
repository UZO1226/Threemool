"""Replay supported cuts and remove only upward deviations in subsequent commands."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e88'))
import selection as S

STEPS = 24
HORIZONS = np.array([1, 4, 12, 24])
TOL = 1e-8


def simulate(rows, archive, keys, q, agent, dyn, cut, tape=None):
    windows = {i: {k: a[t:t+STEPS+1] for k,a in rows[d].items()} for i,(d,t,_) in enumerate(keys)}
    env = S.E.BatchEnv(windows, dyn)
    day, t, member = keys.T
    ep = day*5+member
    env.reset(list(enumerate(member)))
    env.state = archive['states'][ep,t-1].copy()
    env.prev_air = archive['air'][ep,t-1].copy()
    env.prev_zone = env.prev_air[:,None]*np.array([rows[d]['frac'][s-1] for d,s in zip(day,t)])
    obs = env.obs()
    actions = []
    metrics = []
    for k in range(STEPS):
        if tape is not None:
            action = tape[:,k]
        elif k == 0:
            action = S.R.air_to_action(q-S.R.STEP*cut)
        else:
            action = agent.act_det(obs)[:,0]
        obs, _, cost, done, info = env.step(action)
        if k == 0 and cut:
            np.testing.assert_allclose(env.state, archive['states'][ep,t], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(cost, info['v_nh4']+info['v_do'])
        metrics.append(np.column_stack((info['air']*.25, info['v_nh4'], info['v_do'],
                                        np.maximum(env.state[:,0]-4, 0)*.25)))
        actions.append(action)
    assert done
    return np.stack(metrics,axis=1), np.stack(actions,axis=1)


def replay(rows, archive, keys, q, agent, dyn, cap, smoke=False):
    outputs = []
    for start in range(0,len(keys),250):
        b = slice(start,start+250)
        args = (rows, archive, keys[b], q[b], agent, dyn)
        a, control = simulate(*args, False)
        b_metrics, cut = simulate(*args, True)
        metrics = [a, b_metrics]
        np.testing.assert_allclose(a[:,0,0]-b_metrics[:,0,0],125,atol=TOL,rtol=0)
        if cap:
            c, _ = simulate(*args, True, np.minimum(control,cut))
            np.testing.assert_array_equal(c[:,0], b_metrics[:,0])
            np.testing.assert_allclose(b_metrics[:,:,0]-c[:,:,0],np.maximum(0,b_metrics[:,:,0]-a[:,:,0]),
                                       atol=TOL,rtol=0)
            metrics.append(c)
        if smoke:
            repeat,_ = simulate(*args,True,cut)
            np.testing.assert_array_equal(repeat,b_metrics)
        outputs.append(np.stack(metrics,axis=1))
    return np.concatenate(outputs)


def rates(weights, arr):
    return np.average(arr,weights=weights,axis=0).tolist()


def summary(keys, values, neighbor):
    a,b,c = values.transpose(1,0,2,3)
    delta = b-a
    benefit = b-c
    lost = delta[:,:,0] >= -TOL
    hit = np.maximum.accumulate(lost, axis=1)
    w = S.weights(keys[:,0])
    h = HORIZONS-1
    past_lost = neighbor[:,:,0] >= -TOL
    result = dict(n=len(keys), days=len(set(keys[:,0])), cut_minus_base=rates(w,delta[:,h]),
                  lost_now=rates(w,lost[:,h]), lost_ever=rates(w,hit[:,h]),
                  mismatch=rates(w,(past_lost!=lost)[:,h]))
    # Condition on observed six-hour outcomes only for the secondary summary.
    for name,mask in [('all',np.ones(len(keys),bool)), ('lost',lost[:,-1])]:
        if not mask.any():
            result[name]=None
            continue
        weight = S.weights(keys[mask,0])
        nh = benefit[mask,-1][:,[1,3]]
        good = (nh<=TOL).all(1) & (nh < -TOL).any(1)
        bad = (nh>=-TOL).all(1) & (nh > TOL).any(1)
        tied = (np.abs(nh)<=TOL).all(1)
        result[name]=dict(n=int(mask.sum()), days=len(set(keys[mask,0])),
                          extra=rates(weight,benefit[mask][:,h]),
                          nh_classes=rates(weight,np.column_stack((good,bad,tied,~(good|bad|tied)))))
    return result


def run(root,out,smoke):
    torch.set_num_threads(1)
    dates, rows = S.E.load_train()
    dyn = S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    hashes = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(S.digest(Path(p))==v for p,v in hashes.items())
    refs=json.loads((HERE.parent/'e89/summary.json').read_text())
    e86=json.loads((HERE.parent/'e86/summary.json').read_text())
    records=[]
    if not smoke:
        out.mkdir(parents=True,exist_ok=False)
    for ref in refs:
        seed=ref['seed']
        agent=S.E.restore(seed)
        before=S.E.S.fingerprint(agent)
        paths=[root/'e88/main'/f'{seed}_early_down.npz',root/'e78/main'/f'{seed}_train48_down.npz',
               root/'e88/main'/f'probe{seed}.npz']
        assert [S.digest(p) for p in paths]==ref['input_sha256']
        match_source=root/'e89/main'/f'matches{seed}.npz'
        with np.load(match_source) as match, np.load(paths[2]) as old:
            keys, train_keys = old['keys'], old['train_keys']
            np.testing.assert_array_equal(match['keys'], keys)
            np.testing.assert_array_equal(match['train_keys'], train_keys)
            neighbor, supported = match['ids'][0,:,0], match['supported']
            expected_train=old['train_totals']
        archive, sparse=S.read_candidates(paths[1], dates[140:])
        np.testing.assert_array_equal(sparse,keys)
        probe=root/'e85/main'/f'probe{seed}.npz'
        assert S.digest(probe)==e86[seed]['score_sha256']
        with np.load(probe) as old:
            full,q,expected = old['keys'],old['q'],old['totals']
        subset=S.headroom.nonoverlap(full)
        np.testing.assert_array_equal(full[subset],keys)
        if smoke:
            result=replay(rows[140:],archive,full[:5],q[:5],agent,dyn,True,True)
            np.testing.assert_allclose(result[:,:2].cumsum(2)[:,:,[3,11,23]],expected[:5],atol=1e-7,rtol=0)
            print('replay, cut and subtraction smoke passed')
            return
        query=replay(rows[140:],archive,full,q,agent,dyn,True).cumsum(2)
        np.testing.assert_allclose(query[:,:2][:,:,[3,11,23]],expected,atol=1e-7,rtol=0)
        query=query[subset]
        train_archive, past=S.read_candidates(paths[0],dates[:140])
        np.testing.assert_array_equal(train_keys,past)
        d,t,m=past.T
        flow=train_archive['raw'][d*5+m,t]
        library=replay(rows[:140],train_archive,past,flow,agent,dyn,False).cumsum(2)
        np.testing.assert_allclose(library[:,:,-1],expected_train,atol=1e-7,rtol=0)
        peer=library[neighbor,1]-library[neighbor,0]
        data={g:summary(keys[ix],query[ix],peer[ix]) for g,ix in
              [('supported',supported),('all',np.ones(len(keys),bool))]}
        for g in data:
            np.testing.assert_allclose(data[g]['mismatch'][-1],ref['metrics'][g]['current']['component_disagreement'][0],atol=1e-12)
        assert S.E.S.fingerprint(agent)==before
        np.savez_compressed(out/f'paths{seed}.npz',keys=keys,supported=supported,cumulative=query,
                            peer_delta=peer,peer_keys=past[neighbor])
        records.append(dict(seed=seed,match_sha256=S.digest(match_source),groups=data))
        print(seed, data['supported']['n'], data['supported']['lost_now'], flush=True)
    (out/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path,default=S.ROOT)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    run(args.root,args.out,args.smoke)
