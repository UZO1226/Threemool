"""Run the checked single-head severity candidate through the common trainer."""
import json
from pathlib import Path
from unittest.mock import patch

from severity_agent import SeverityFCSRL,SeverityInfoEnv
from ppo_fcsrl import RepresentationConfig
from train_dw import make_agent
import train_policy

HERE=Path(__file__).resolve().parent

def factory(args,config):
    if args.method!='ppolag_fcsrl_severity':return make_agent(args,config)
    representation=RepresentationConfig()
    return SeverityFCSRL(11,1,config=config,repr_config=representation),representation

if __name__=='__main__':
    for name in ('severity_check.json','environment_check.json'):
        checks=json.loads((HERE/name).read_text())
        assert not checks['test_evaluated'] and not checks['policy_performance_tested']
    assert json.loads((HERE/'environment_check.json').read_text())['all_original_outputs_exact']
    # Scope this runtime factory substitution to this training process. The
    # common evaluator still uses quantile_env.FrozenQuantileEnv, unmodified.
    # No frozen file, transition implementation or installed package is edited.
    with patch.object(train_policy,'FrozenQuantileEnv',SeverityInfoEnv):
        train_policy.main(agent_factory=factory,extra_methods=('ppolag_fcsrl_severity',))
