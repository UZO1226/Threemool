"""Run the split-supervision hypothesis and its capacity control."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e42'))
from split_feasibility import TwoHeadFCSRL
from ppo_fcsrl import RepresentationConfig
from train_dw import make_agent
from train_policy import main


def factory(args,config):
    modes={'ppolag_fcsrl_split':'split','ppolag_fcsrl_capacity':'capacity'}
    if args.method not in modes:return make_agent(args,config)
    r=RepresentationConfig()
    return TwoHeadFCSRL(11,1,config=config,repr_config=r,mode=modes[args.method]),r


if __name__=='__main__':main(agent_factory=factory,extra_methods=('ppolag_fcsrl_split','ppolag_fcsrl_capacity'))
