"""e58 학습 전 경계, 기본 경로 보존, 경사 연결, GPU 저장/재개 검사."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import severity as s
base, h = s.base, s.h


def hashes():
    paths = list(s.HERE.glob('*.py')) + [s.HERE/'protocol.json']
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and path.endswith('.py') and ('/Yoo/' in path or '/FCSRL/fcsrl/' in path):
            paths.append(Path(path))
    prepared = Path(json.loads((s.HERE/'protocol.json').read_text())['prepared'])
    paths += [prepared/n for n in ['data.pkl','surrogate.pkl','normalizer.json','prior.pt']]
    return {str(p.resolve()):base.sha(p) for p in set(paths)}


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    c = base.common.config(1)
    prepared = Path(json.loads((s.HERE/'protocol.json').read_text())['prepared'])
    constant = np.full((190, 2), 8.)
    end = np.zeros_like(constant)
    end[[94,189]] = 1
    y, valid = s.targets(constant, end)
    assert valid.sum() == 92*2*2
    assert np.all(y[valid.astype(bool)] == 1)
    assert valid[91].all() and not valid[92:95].any() and valid[95].all()
    toy = np.arange(190*2).reshape(190, 2)/10
    y, valid = s.targets(toy, end)
    for t,n in zip(*np.where(valid)):
        assert not end[t:t+3,n].any()
        np.testing.assert_allclose(y[t,n],np.maximum(toy[t:t+4,n]-4,0).mean()/4,rtol=1e-6)
    # 그룹 수가 달라도 같은 비중, 단일 그룹과 유효 표본 0 경계를 확인한다.
    assert s.balanced_mse(torch.tensor([2.,0.,0.,0.]),torch.tensor([1.,0.,0.,0.]),torch.ones(4)).item()==.5
    assert s.balanced_mse(torch.zeros(2),torch.zeros(2),torch.zeros(2)).item()==0
    assert s.balanced_mse(torch.zeros(2),torch.ones(2),torch.ones(2)).item()==1
    splits, days = base.load_selected(prepared/'data.pkl')
    dyn = base.Dynamics(prepared/'surrogate.pkl')
    baseline = s.make_agent('baseline',0,prepared,c)
    initial = base.fingerprint(baseline)
    initial_rng = torch.get_rng_state()
    for method in ['severity','detached']:
        agent = s.make_agent(method,0,prepared,c)
        assert base.fingerprint(agent)==initial
        assert torch.equal(torch.get_rng_state(),initial_rng)
    baseline = s.make_agent('baseline',0,prepared,c)
    original, od = base.e30.collect(baseline,c,base.BatchEnv(days,dyn),splits['train'],np.random.default_rng(0))
    baseline = s.make_agent('baseline',0,prepared,c)
    buf, bd = s.collect(baseline,c,s.RecordingEnv(days,dyn),splits['train'],np.random.default_rng(0))
    for key in original:
        np.testing.assert_array_equal(original[key],buf[key])
    assert all(od[k]==bd[k] for k in od)
    assert bd['severity_valid_fraction']==92/95
    post_collect_rng = torch.get_rng_state()
    initial_hashes = hashes()
    result = dict(passed=False,checks=dict(target_boundary=True,balanced_loss=True,
        same_initial_weights_and_rng=True,original_collection_exact=True),collection=bd,updates={})
    policies = {}
    for method in s.METHODS:
        learner, collector = s.make_pair(method,0,prepared,c)
        torch.set_rng_state(post_collect_rng)
        with h.gpu_slot('cuda:0'):
            diagnostic = s.gradient_diagnostics(learner,buf)
        tick = time.monotonic()
        info = h.update(learner,buf)
        h.sync_collector(learner,collector)
        policies[method]=base.fingerprint(collector)
        result['updates'][method]=dict(info=info,diagnostics=diagnostic,seconds=time.monotonic()-tick)
        h.configure('cpu')
        dest = out/method
        collector.save(dest)
        restored=s.make_agent(method,0,prepared,c)
        restored.load(dest)
        assert s.fingerprint(restored)==s.fingerprint(collector)
        if method=='severity':
            assert diagnostic['aux_gradient_norm_ratio']>0
            rng=np.random.default_rng(5800)
            metadata=dict(check='e58_severity_resume')
            learner.pid.update(20.,12.);learner.lagrg=learner.pid.get_lagrg()
            h.checkpoint(out/'resume.pt',learner,rng,[],metadata)
            uninterrupted=h.update(learner,buf)
            expected=s.fingerprint(learner); expected_rng=torch.get_rng_state()
            resumed,_=s.make_pair(method,0,prepared,c)
            h.restore(out/'resume.pt',resumed,np.random.default_rng(9),metadata)
            continued=h.update(resumed,buf)
            assert s.fingerprint(resumed)==expected and uninterrupted==continued
            assert torch.equal(torch.get_rng_state(),expected_rng)
        if method=='detached':
            assert diagnostic['aux_gradient_norm_ratio']==0
    assert policies['baseline']==policies['detached']
    assert policies['baseline']!=policies['severity']
    assert all(base.sha(p)==v for p,v in initial_hashes.items())
    result['checks'].update(detached_full_update_matches_baseline=True,severity_changes_encoder=True,
        checkpoint_roundtrip=True,next_update_resume_exact=True,hashes_unchanged=True)
    result.update(passed=True,hashes=initial_hashes,elapsed_seconds=time.monotonic()-started,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
        test_evaluated=False)
    base.save(out/'checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    check(p.parse_args().out)
