"""Preflight accounting, candidate ordering and unmodified-environment parity."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import plan75 as P


def main(out):
    t0 = time.monotonic()
    ctx = P.context()
    reports = {}
    for split in ('dev','val'):
        env = P.env_for(ctx)
        actual, *_ = P.E.rollout(env, ctx[0][split], P.B.ABAC(env, **ctx[4]['params']))
        for key, expected in ctx[4][split].items():
            if expected is not None:
                np.testing.assert_allclose(actual[key], expected, rtol=1e-10, atol=1e-8, err_msg=key)
        reports[split] = dict(baseline_aggregate_parity=True, air_m3=actual['air_m3'], cost=actual['step_cost'])
    eps = P.E.episodes(ctx[0]['dev'][:2],ctx[2].n_members)
    a, st = P.abac_actions(ctx, eps)
    sm, hist = P.simulate(ctx, eps, a[:,None,:],trace=True)
    np.testing.assert_array_equal(hist['states'],st)
    acts = np.stack([a, np.clip((a+1)*.95-1,-1,1), np.full_like(a,-1)],1)
    tick = time.monotonic()
    vm, _ = P.simulate(ctx,eps,acts)
    for k in range(3):
        individual,_ = P.simulate(ctx,eps,acts[:,k:k+1,:])
        for key in vm:
            np.testing.assert_allclose(vm[key][:,k],individual[key][:,0],rtol=0,atol=1e-8)
    baseline = {key:v[:,0] for key,v in sm.items()}
    for mode in ('air','cost'):
        order, feasible = P.rank(vm,baseline,mode)
        assert feasible[np.arange(len(eps)),order[:,0]].all()
    # Costs at exactly the thresholds must be zero; either strict exceedance is one.
    np.testing.assert_array_equal(P.V.step_cost(np.array([4,4.01,3,4.01]),np.array([380,380,381,381])),[0,1,1,1])
    fake = {key:np.array([[1.,1.,1.]]) for key in P.KEYS}
    fb = {key:np.array([1.]) for key in P.KEYS}
    fake['air_m3'][0] = [.8,.7,1.]
    fake['step_cost'][0] = [1.,2.,1.]
    for mode in ('air','cost'):
        order,feasible=P.rank(fake,fb,mode)
        assert order[0,0]==0 and not feasible[0,1]
    # Actual full population cost on a fixed development-only smoke, no selection or tuning.
    pop=np.repeat(a[:,None,:],P.POP,axis=1)
    tick=time.monotonic();pm,_=P.simulate(ctx,eps,pop);pop_seconds=time.monotonic()-tick
    for key in sm:
        np.testing.assert_allclose(pm[key],np.repeat(sm[key],P.POP,axis=1),rtol=0,atol=1e-8)
    reports.update(action_replay=True,candidate_layout=True,threshold_cost=True,feasible_rank=True,
                   population32_two_dates_seconds=pop_seconds,total_seconds=time.monotonic()-t0)
    P.write_json(out,reports)
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
