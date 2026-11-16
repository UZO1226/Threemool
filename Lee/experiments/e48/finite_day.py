"""Finite-day feasibility supervision, preserving PPO/PID and actual costs."""
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


class FiniteDayFCSRL(PPOFCSRLAgent):
    def finish_buffer(self,buf):
        # Only this auxiliary recurrence treats the observed episode end as
        # its horizon. Preserve the actual terminated/end arrays used by PPO.
        auxiliary_terminal=np.maximum(buf['terminated'],buf['end'])
        buf['feasi']=feasibility_targets(buf['cost'],buf['next_feasi'],buf['end'],
            auxiliary_terminal,self.repr_config.feasibility_discount)

    def save(self,path):
        super().save(path)
        (Path(path)/'finite_day.json').write_text(json.dumps({'auxiliary_horizon':'observed episode end',
            'ppo_bootstrap_unchanged':True,'actual_environment_cost_unchanged':True},indent=2))

    def load(self,path):
        metadata=json.loads((Path(path)/'finite_day.json').read_text())
        assert metadata['auxiliary_horizon']=='observed episode end'
        assert metadata['ppo_bootstrap_unchanged'] and metadata['actual_environment_cost_unchanged']
        super().load(path)
