"""Evaluate final seed-paired checkpoints, with quality-blind air matching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e42'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e43'))
sys.path.insert(0,str(ROOT/'Lee/experiments/e45'))
from split_feasibility import TwoHeadFCSRL
from severity_agent import SeverityFCSRL
from quantile_env import FrozenDynamics,evaluate,WORKSPACE,frozen_digest
from ppo_fcsrl import PPOFCSRLAgent,RepresentationConfig
from ppo_lag import PPOLagAgent
from ppo_config import PPOConfig
from surrogate_dw import GBMDynamics
from air_match import match_air


def file_hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob('*')) if p.is_file()}


def freeze_identities(path):
    raw=path.read_bytes();normalized=raw.replace(b'\r\n',b'\n')
    return {hashlib.sha256(b).hexdigest() for b in (raw,normalized,normalized.replace(b'\n',b'\r\n'))}


def load_run(folder,freeze_path,allow_smoke=False):
    config=json.loads((folder/'effective_config.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text())
    progress=json.loads((folder/'progress.json').read_text());assert progress['phase']=='complete'
    metadata=config['experiment_metadata'];assert not metadata['test_evaluated']
    manifest=json.loads((folder.parent.parent/'manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():
        path=Path(name)
        if not path.exists():
            relative=name.replace('\\','/').split('/Threemool/',1)[1];path=ROOT/relative
        assert digest in freeze_identities(path),f'Training source changed: {name}'
    assert metadata['environment_freeze_sha256'] in freeze_identities(freeze_path)
    assert metadata['terminal_reward_air_alias_corrected']
    c=PPOConfig.from_dict(config['config'])
    assert c.cost_limit==12 and c.n_env==8 and c.horizon==760
    assert metrics['steps']==c.actual_steps
    assert c.actual_steps==(6080 if allow_smoke else 401280)
    method=config['method'];r=RepresentationConfig(**config['representation']) if config['representation'] else None
    torch.manual_seed(0)
    if method=='ppolag':agent=PPOLagAgent(11,1,config=c)
    elif method in ('ppolag_fcsrl','ppolag_fcsrl_cold'):agent=PPOFCSRLAgent(11,1,config=c,repr_config=r)
    elif method=='ppolag_fcsrl_severity':agent=SeverityFCSRL(11,1,config=c,repr_config=r)
    else:
        modes={'ppolag_fcsrl_split':'split','ppolag_fcsrl_capacity':'capacity'}
        agent=TwoHeadFCSRL(11,1,config=c,repr_config=r,mode=modes[method])
    checkpoint=folder/'model_final';hashes=file_hashes(checkpoint);agent.load(checkpoint)
    return agent,config,metrics,hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--environment',choices=['old','new'],default='new')
    p.add_argument('--kind',choices=['agent','capacity','environment','diagnostic'],default='agent')
    p.add_argument('--allow-smoke',action='store_true');p.add_argument('--native-only',action='store_true')
    p.add_argument('--baseline-evaluation',type=Path);p.add_argument('--candidate-evaluation',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);os.environ['DARROW_FAST_DYNAMICS']='1'
    freeze=json.loads(a.freeze.read_text())
    expected=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
    assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected
    baseline,bc,bm,bhash=load_run(a.baseline,a.freeze,a.allow_smoke)
    candidate,cc,cm,chash=load_run(a.candidate,a.freeze,a.allow_smoke)
    assert bc['seed']==cc['seed'];assert bc['config']==cc['config']
    if a.kind in ('agent','capacity'):
        assert a.environment=='new'
        assert bc['experiment_metadata']['environment']==cc['experiment_metadata']['environment']=='new'
        assert bc['representation']==cc['representation']
        if a.kind=='agent':assert bc['method']=='ppolag_fcsrl' and cc['method'] in ('ppolag_fcsrl_split','ppolag_fcsrl_capacity','ppolag_fcsrl_severity','ppolag_fcsrl_cold')
        else:assert bc['method']=='ppolag_fcsrl_capacity' and cc['method']=='ppolag_fcsrl_split'
    if a.kind=='environment':
        assert bc['method']==cc['method']=='ppolag_fcsrl'
        assert bc['experiment_metadata']['environment']=='old' and cc['experiment_metadata']['environment']=='new'
    original=WORKSPACE/freeze['original_environment_path']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==freeze['original_environment_sha256']
    dynamics=FrozenDynamics(a.freeze) if a.environment=='new' else GBMDynamics(original)
    def score(agent,offset,label):
        summary,rows=evaluate(a.data,dynamics,act_batch=lambda x:agent.act_det(x)[:,0],air_offset=offset)
        (a.out/f'{label}.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2,allow_nan=False))
        return summary,rows
    def native(agent,config,hashes,cached,label):
        if cached is None:return score(agent,0,label)
        stored=json.loads(cached.read_text())
        assert stored['checkpoint_hashes']==hashes and stored['policy_seed']==config['seed']
        assert stored['method']==config['method'] and stored['evaluation_environment']==a.environment
        assert stored['freeze_canonical_sha256']==frozen_digest(a.freeze) and stored['split']=='val'
        assert not stored['test_evaluated'] and not stored['smoke_only'] and stored['training_steps']==401280
        assert stored['checkpoint_loaded_and_evaluated']
        (a.out/f'{label}.json').write_text(json.dumps({'summary':stored['summary'],'rows':stored['rows']},indent=2))
        return stored['summary'],stored['rows']
    bscore,brows=native(baseline,bc,bhash,a.baseline_evaluation,'baseline_native')
    cscore,crows=native(candidate,cc,chash,a.candidate_evaluation,'candidate_native')
    for config,metrics,summary in [(bc,bm,bscore),(cc,cm,cscore)]:
        if config['experiment_metadata']['environment']==a.environment:
            for key,value in metrics['val_final'].items():np.testing.assert_allclose(summary[key],value,rtol=1e-10,atol=1e-10)
    matching=None;matched=None
    if not a.native_only:
        cache={0.:(cscore,crows)}
        def air_only(offset):
            result=score(candidate,offset,f'query_{len(cache):02d}');cache[offset]=result
            print('AIR_QUERY',offset,result[0]['air_m3'],flush=True)
            return result[0]['air_m3']
        matching=match_air(air_only,bscore['air_m3'],cscore['air_m3'])
        offset=matching['offset']
        if offset not in cache:cache[offset]=score(candidate,offset,'matched_endpoint')
        matched,matched_rows=cache[offset]
        np.testing.assert_allclose(matched['air_m3'],matching['achieved_air_m3'],rtol=1e-10,atol=1e-8)
        (a.out/'candidate_matched.json').write_text(json.dumps({'summary':matched,'rows':matched_rows},indent=2))
    assert file_hashes(a.baseline/'model_final')==bhash
    assert file_hashes(a.candidate/'model_final')==chash
    result={'policy_seed':bc['seed'],'kind':a.kind,'evaluation_environment':a.environment,
        'baseline_method':bc['method'],'candidate_method':cc['method'],
        'baseline_training_environment':bc['experiment_metadata']['environment'],
        'candidate_training_environment':cc['experiment_metadata']['environment'],
        'baseline_native':bscore,'candidate_native':cscore,'candidate_matched':matched,'matching':matching,
        'freeze_canonical_sha256':frozen_digest(a.freeze),'baseline_checkpoint_hashes':bhash,
        'candidate_checkpoint_hashes':chash,'training_steps':bm['steps'],'test_evaluated':False,
        'smoke_only':a.allow_smoke,'checkpoint_replay_verified':True,
        'scope':'Exploratory reused validation, frozen surrogate, final policies. Matching changes candidate actions using air only.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('PAIR_COMPLETE',bc['seed'],a.kind,json.dumps({'native':cscore,'matched':matched}),flush=True)


if __name__=='__main__':main()
