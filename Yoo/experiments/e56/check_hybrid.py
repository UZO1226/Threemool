"""동일 수집/미니배치, CPU/GPU 수치 근접성 및 optimizer 포함 재개를 검사한다."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
import train as base
import hybrid as h


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    prepared=Path('/home/jio/fcsrl_src/results/e50/prepared')
    splits,days=base.load_selected(prepared/'data.pkl')
    env=base.BatchEnv(days,base.Dynamics(prepared/'surrogate.pkl'))
    result={}
    for method in ['ppo','fcsrl']:
        c=base.common.config(2);device='cpu' if method=='ppo' else 'cuda:0'
        h.configure('cpu');reference=base.make_agent(method,0,prepared,c,'off')
        learner,collector=h.make_pair(method,0,prepared,c,'off',device)
        assert base.fingerprint(reference)==base.fingerprint(collector)
        h.sync_collector(learner,collector)
        assert base.fingerprint(learner)==base.fingerprint(collector)
        h.configure('cpu');sampler=np.random.default_rng(0)
        t=time.monotonic();buf,diag=base.e30.collect(collector,c,env,splits['train'],sampler);collect=time.monotonic()-t
        state=torch.get_rng_state()
        torch.set_rng_state(state);h.configure('cpu');tick=time.monotonic()
        ri=reference.update(buf);cpu_seconds=time.monotonic()-tick
        cpu_rng_after=torch.get_rng_state()
        torch.set_rng_state(state);tick=time.monotonic();gi=h.update(learner,buf);device_seconds=time.monotonic()-tick
        assert torch.equal(torch.get_rng_state(),cpu_rng_after)
        h.sync_collector(learner,collector);h.configure('cpu')
        probe=buf['obs'][:1024]
        with torch.no_grad():
            a=reference.dist(reference._t(probe)).mean;b=collector.dist(collector._t(probe)).mean
            error=float(abs(a-b).max())
            vr0,vc0=reference.values(reference._t(probe));vr1,vc1=collector.values(collector._t(probe))
            value_error=max(float(abs(vr0-vr1).max()),float(abs(vc0-vc1).max()))
        assert ri['pi_epochs']==gi['pi_epochs']
        assert error<1e-3 and value_error<2e-3,(method,error,value_error)
        if method=='ppo':assert base.fingerprint(reference)==base.fingerprint(learner)
        learner.pid.update(diag['daily_cost_mean'],12);learner.lagrg=learner.pid.get_lagrg()
        meta=dict(method=method,device=device,test='resume')
        h.checkpoint(out/f'{method}_resume.pt',learner,sampler,[{'update':1}],meta)
        expected_sampler=copy.deepcopy(sampler.bit_generator.state)
        saved_fp=base.fingerprint(learner);saved_pid=vars(learner.pid).copy()
        # 연속된 다음 갱신과 checkpoint에서 복원한 다음 갱신을 대조한다.
        ni=h.update(learner,buf);expected_fp=base.fingerprint(learner);expected_rng=torch.get_rng_state()
        restored,_=h.make_pair(method,999,prepared,c,'off',device)
        next_sampler=np.random.default_rng(999)
        logs=h.restore(out/f'{method}_resume.pt',restored,next_sampler,meta)
        assert logs==[{'update':1}]
        assert base.fingerprint(restored)==saved_fp and vars(restored.pid)==saved_pid
        assert next_sampler.bit_generator.state==expected_sampler
        rni=h.update(restored,buf)
        assert base.fingerprint(restored)==expected_fp and ni==rni
        assert torch.equal(torch.get_rng_state(),expected_rng)
        result[method]=dict(cpu_threads=torch.get_num_threads(),device=device,collection_seconds=collect,
            cpu1_update_seconds=cpu_seconds,hybrid_update_seconds=device_seconds,
            mean_action_max_difference=error,value_max_difference=value_error,same_epochs=gi['pi_epochs'],
            cpu_rng_identical=True,checkpoint_optimizer_rng_resume_exact=True,finite=True)
        print('PASS',method,result[method],flush=True)
    assert torch.get_num_threads()==1
    hashes={str(p):base.sha(p) for p in [Path(__file__),Path(h.__file__),Path(base.__file__)]}
    base.save(out/'checks.json',dict(passed=True,checks=result,hashes=hashes,
        caveat='Fixed buffer and initial update parity is not proof of identical full CPU/GPU training trajectories'))
    print('HYBRID CHECKS PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
