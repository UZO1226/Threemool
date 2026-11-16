"""Use unchanged FCSRL networks with a checked colder-day training sampler."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import sys

from cold_sampling import ColdTrainingEnv

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e42'))
import train_policy
from ppo_fcsrl import PPOFCSRLAgent,RepresentationConfig
from train_dw import make_agent

def factory(args,config):
    if args.method!='ppolag_fcsrl_cold':return make_agent(args,config)
    representation=RepresentationConfig()
    return PPOFCSRLAgent(11,1,config=config,repr_config=representation),representation

if __name__=='__main__':
    checks=json.loads(Path(__file__).with_name('sampling_check.json').read_text())
    assert checks['real_environment_step_outputs_exact'] and checks['training_days_only']
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path,required=True)
    known,_=parser.parse_known_args()
    class RecordedColdEnv(ColdTrainingEnv):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            (known.out/'sampling.json').write_text(json.dumps(self.sampling_info,indent=2))
    with patch.object(train_policy,'FrozenQuantileEnv',RecordedColdEnv):
        train_policy.main(agent_factory=factory,extra_methods=('ppolag_fcsrl_cold',))
