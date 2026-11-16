"""Prepared severity-supervised FCSRL candidate, preserving actual CMDP costs."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e42'))
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
sys.path.insert(0,str(ROOT.parent/'FCSRL'))
import fcsrl_compat
from ppo_fcsrl import PPOFCSRLAgent,feasibility_targets
from quantile_env import FrozenQuantileEnv


def severity_cost(actual_cost,next_state,weight=1.):
    state=np.asarray(next_state)
    magnitude=np.maximum(state[...,0]/4-1,0)+np.maximum(state[...,2:5].max(-1)/2-1,0)
    return np.asarray(actual_cost)+weight*magnitude


class SeverityInfoEnv(FrozenQuantileEnv):
    """Expose already-observed terminal concentrations only to auxiliary labels."""
    def step(self,action):
        obs,reward,cost,done,terminal,info=super().step(action)
        # terminal is captured before reset on every transition. Its first
        # five fields are physical concentrations, before actor normalization.
        info=dict(info,next_state_for_aux=terminal[:,:5].copy())
        return obs,reward,cost,done,terminal,info


class SeverityFCSRL(PPOFCSRLAgent):
    buffer_shapes={'severity_cost':()}

    def __init__(self,*args,severity_weight=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if severity_weight not in (0.,1.):raise ValueError('Only defined candidate or equivalence-check weight')
        self.severity_weight=severity_weight

    def collect_aux(self,buf,t,next_obs,cost,info):
        np.testing.assert_array_equal(cost,info['v_nh4']+info['v_do'])
        buf['severity_cost'][t]=severity_cost(cost,info['next_state_for_aux'],self.severity_weight)

    def finish_buffer(self,buf):
        # The original train_encoder, network, loss coefficients, EMA, actor,
        # reward/cost critics and PID all remain inherited without overrides.
        buf['feasi']=feasibility_targets(buf['severity_cost'],buf['next_feasi'],buf['end'],
            buf['terminated'],self.repr_config.feasibility_discount)

    def save(self,path):
        super().save(path)
        (Path(path)/'severity.json').write_text(json.dumps({'severity_weight':self.severity_weight,
            'actual_environment_cost_unchanged':True,'terminal_concentrations_float32':True},indent=2))

    def load(self,path):
        assert json.loads((Path(path)/'severity.json').read_text())['severity_weight']==self.severity_weight
        super().load(path)
