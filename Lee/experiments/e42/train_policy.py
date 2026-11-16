"""Run unchanged PPO/FCSRL with a selected, reset-safe frozen environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
sys.path.insert(0,str(ROOT.parent/'FCSRL'))
import fcsrl_compat
from train_dw import train
from surrogate_dw import GBMDynamics
from quantile_env import FrozenDynamics,FrozenQuantileEnv,TerminalSafeOriginalEnv,evaluate


def main(agent_factory=None,extra_methods=()):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--environment',choices=['old','new'],required=True)
    p.add_argument('--method',choices=['ppolag','ppolag_fcsrl',*extra_methods],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=400000)
    p.add_argument('--evaluations',type=int,default=5);a=p.parse_args()
    freeze=json.loads(a.freeze.read_text());assert freeze['gate']['passed']
    expected_data=next(v for k,v in freeze['frozen_files_sha256'].items() if k.endswith('darrow_15min.pkl'))
    assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected_data
    original=ROOT.parent/freeze['original_environment_path']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==freeze['original_environment_sha256']
    check=json.loads((a.freeze.parent/'environment_check.json').read_text())
    assert check['full_day_forecast_adapter_agreement'] and check['test_split_rejected']
    reset=json.loads((a.freeze.parent/'reset_alias_check.json').read_text());assert reset['all95_corrected_rewards_pass']
    os.environ['DARROW_FAST_DYNAMICS']='1'
    upstream=ROOT.parent/'FCSRL'
    upstream_commit=subprocess.check_output(['git','-c',f'safe.directory={upstream.as_posix()}','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    assert upstream_commit=='d882b91054d52c7a63a18a21f7db67bbbed82b63'
    assert not subprocess.check_output(['git','-c',f'safe.directory={upstream.as_posix()}','status','--porcelain','--untracked-files=no'],cwd=upstream,text=True).strip()
    dyn=FrozenDynamics(a.freeze) if a.environment=='new' else GBMDynamics(original)
    cls=FrozenQuantileEnv if a.environment=='new' else TerminalSafeOriginalEnv
    args=SimpleNamespace(method=a.method,seed=a.seed,steps=a.steps,evaluations=a.evaluations,
        limit=12.,data=a.data,surrogate=dyn.path if a.environment=='new' else original,
        out=a.out,pid_mult=1.,fpen_weight=.1)
    def evaluator(data,dynamics,split,act,limit):
        assert limit==12
        return evaluate(data,dynamics,act_batch=act,split=split,draws=8)
    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata={'code_commit':commit,'environment':a.environment,'environment_freeze_sha256':hashlib.sha256(a.freeze.read_bytes()).hexdigest(),
        'upstream_commit':upstream_commit,'terminal_reward_air_alias_corrected':True,'policy_seed':a.seed,'test_evaluated':False,
        'primary_checkpoint':'final','evaluation_noise_draws':8 if a.environment=='new' else 1,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path(__file__).parent.glob('*.py'),*(ROOT/'Lee/code/darrow').glob('*.py')]}}
    train(args,dynamics=dyn,environment_factory=lambda data,dynamics,n,seed:cls(data,dynamics,n_env=n,seed=seed),
        evaluation_function=evaluator,agent_factory=agent_factory,run_metadata=metadata)


if __name__=='__main__':main()
