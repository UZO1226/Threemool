"""Test whether the frozen action-conditioned head ranks harmful supported cuts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e80'))
import pulse as P

E = P.E
R = P.prior
SOURCE = Path('/home/jio/fcsrl_src/results/e78/main')


@torch.no_grad()
def head(agent, obs, q):
    state = agent._t(agent.obs_normalizer(obs))
    act = agent._t(R.air_to_action(q)[:,None])
    z = agent.encoder.zsa(agent.encoder.zs(state),act)
    logits = agent.feasi_head(z)
    assert logits.shape == (len(obs),agent.n_buckets)
    scores = E.S.DiscDist(logits.unsqueeze(0),agent.bucket_low,agent.bucket_high,agent.n_buckets).mean()
    return scores.cpu().numpy().reshape(-1).astype(float)


def train_threshold(agent, tree, rows, support):
    keys = np.array([(d,t) for d in range(140) for t in P.TIMES])
    obs = R.observations(rows,keys)
    q = R.action_to_air(agent.act_det(obs)[:,0])
    take = R.command(obs,q,tree,support,'down') == -1
    x, flow = obs[take], q[take]
    scores = head(agent,x,flow-R.STEP)-head(agent,x,flow)
    return keys[take], scores, float(np.median(scores))


def validation(archive,rows,agent,tree,support,dyn):
    ep,col = np.where(archive['direction'][:,P.TIMES] == -1)
    t = P.TIMES[col]
    day,member = ep//5,ep%5
    keys = np.column_stack((day,t,member))
    q = archive['raw'][ep,t]
    obs = R.observations(rows,keys[:,:2])
    before = archive['states'][ep,t-1]
    prev_air = archive['air'][ep,t-1]
    obs[:,:5] = before
    obs[:,10] = prev_air/9000
    np.testing.assert_allclose(R.action_to_air(agent.act_det(obs)[:,0]),q,rtol=0,atol=.02)
    np.testing.assert_array_equal(R.command(obs,q,tree,support,'down'),-np.ones(len(q)))
    normal,cut = head(agent,obs,q),head(agent,obs,q-R.STEP)
    exo = np.array([rows[d]['exo'][step] for d,step in zip(day,t)])
    prev_zone = prev_air[:,None]*np.array([rows[d]['frac'][step-1] for d,step in zip(day,t)])
    issued = R.action_to_air(R.air_to_action(q-R.STEP))
    zone = issued[:,None]*np.array([rows[d]['frac'][step] for d,step in zip(day,t)])
    np.testing.assert_allclose(dyn.step(before,exo,zone,prev_zone,member),archive['states'][ep,t],rtol=0,atol=1e-10)
    totals = np.empty((len(keys),2,3,4))
    for start in range(0,len(keys),250):
        b = slice(start,start+250)
        for i,delta in enumerate([0.,-R.STEP]):
            totals[b,i] = P.branch(rows,dyn,agent,keys[b],q[b],delta,archive)
    return keys,q,normal,cut,totals


def stats(keys,score,threshold,totals):
    day = keys[:,0]
    dates,idx,counts = np.unique(day,return_inverse=True,return_counts=True)
    w = 1/counts[idx]
    change = totals[:,1,-1] - totals[:,0,-1]
    harm = (change[:,1]>=1)|(change[:,3]>=.1)
    veto = score>threshold
    def rate(mask):
        return float(np.average(harm[mask],weights=w[mask])) if mask.any() else None
    auc = float(roc_auc_score(harm,score,sample_weight=w)) if len(np.unique(harm))==2 else None
    rates = [rate(mask) for mask in (~veto,veto)]
    class_days = [int(len(np.unique(day[mask]))) for mask in (~harm,harm)]
    passed = (min(class_days)>=5 and auc is not None and auc>=.6 and None not in rates and rates[1]>rates[0])
    return dict(n=len(keys),dates=len(dates),class_days=class_days,positive=int(harm.sum()),
                blocked=int(veto.sum()),allowed_rate=rates[0],blocked_rate=rates[1],auc=auc,passed=bool(passed),
                all_rate=rate(np.ones(len(day),bool)),delta=np.average(change,axis=0,weights=w).tolist())


def main(out,smoke):
    torch.set_num_threads(1)
    hashes=json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    tree_hashes=json.loads((HERE.parent/'e79/summary.json').read_text())['e78_input_sha256']
    dates,rows=E.load_train()
    dyn=E.Dynamics(E.DATA/'surrogate.pkl')
    support=R.fit_support(rows[:140])
    if smoke:
        agent=E.restore(0)
        x=R.observations(rows,[(0,4),(1,8)])
        q=R.action_to_air(agent.act_det(x)[:,0])
        np.testing.assert_array_equal(head(agent,x,q)-head(agent,x,q),0)
        logits=agent.feasi_head(agent.encoder.zsa(agent.encoder.zs(agent._t(agent.obs_normalizer(x))),
                                                 agent._t(R.air_to_action(q)[:,None])))
        buckets=torch.linspace(agent.bucket_low,agent.bucket_high,agent.n_buckets)
        expectation=(logits.softmax(-1)*buckets).sum(-1)
        manual=(expectation.sign()*expectation.abs().expm1()).detach().numpy()
        np.testing.assert_allclose(head(agent,x,q),manual,atol=1e-6,rtol=1e-6)
        keys=np.repeat(np.arange(10)[:,None],2,axis=0)
        labels=np.tile([0.,1.],10)
        totals=np.zeros((20,2,3,4))
        totals[:,1,-1,1]=labels
        assert stats(keys,labels,.5,totals)['passed'] and not stats(keys,-labels,-.5,totals)['passed']
        print('score decoding and harm direction smoke passed')
        return
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for seed in range(5):
        agent=E.restore(seed)
        before=E.S.fingerprint(agent)
        tree_path=SOURCE/f'tree{seed}.joblib'
        assert hashlib.sha256(tree_path.read_bytes()).hexdigest()==tree_hashes[tree_path.name]
        tree=joblib.load(tree_path)
        fit_keys,fit_scores,threshold=train_threshold(agent,tree,rows,support)
        path=SOURCE/f'{seed}_train48_down.npz'
        with np.load(path) as z:
            archive={k:z[k] for k in ['dates','states','air','raw','direction']}
        np.testing.assert_array_equal(archive['dates'],np.asarray(dates[140:],str))
        ref=next(r for r in json.loads((HERE.parent/'e78/records.json').read_text())
                 if r['seed']==seed and r['mode']=='down' and r['split']=='train48')
        assert abs((archive['direction']==-1).sum(1).mean()-ref['down_steps'])<1e-8
        keys,q,normal,cut,totals=validation(archive,rows[140:],agent,tree,support,dyn)
        assert all(np.isfinite(a).all() for a in (totals,normal,cut,fit_scores))
        result=stats(keys,cut-normal,threshold,totals)
        results.append(dict(seed=seed,fit_n=len(fit_scores),threshold=threshold,
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**result))
        np.savez_compressed(out/f'probe{seed}.npz',dates=np.asarray(dates[140:],str),
                            fit_keys=fit_keys,fit_scores=fit_scores,keys=keys,q=q,normal=normal,cut=cut,totals=totals)
        assert E.S.fingerprint(agent)==before
        print(seed,result['n'],result['auc'],result['passed'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(rows=results,advance=sum(r['passed'] for r in results)>=4),
                                              indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    main(args.out,args.smoke)
