"""Train the daily-horizon auxiliary variant through the common runner."""
import json
from pathlib import Path

from finite_day import FiniteDayFCSRL
from ppo_fcsrl import RepresentationConfig
from train_dw import make_agent
from train_policy import main

def factory(args,config):
    if args.method!='ppolag_fcsrl_finite_day':return make_agent(args,config)
    representation=RepresentationConfig()
    return FiniteDayFCSRL(11,1,config=config,repr_config=representation),representation

if __name__=='__main__':
    checks=json.loads(Path(__file__).with_name('finite_check.json').read_text())
    assert checks['ppo_buffer_fields_unchanged'] and checks['hand_computed_daily_targets_pass']
    main(agent_factory=factory,extra_methods=('ppolag_fcsrl_finite_day',))
