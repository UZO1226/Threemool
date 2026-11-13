"""Train-only exposure to colder days without changing transition or reward."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e42'))
from quantile_env import FrozenQuantileEnv


def cold_day_weights(table,days):
    if len(days)<2:raise ValueError('At least two training days required')
    temperature=np.array([float(table.loc[day].temp.mean()) for day in days])
    if not np.isfinite(temperature).all():raise ValueError('Nonfinite training temperature')
    count=int(np.ceil(len(days)*.25))
    selected=np.argsort(temperature,kind='stable')[:count]
    weights=np.full(len(days),.5/(len(days)-count));weights[selected]=.5/count
    return weights,{'selector':'Mean observed training-day temperature, ascending, stable date-order ties',
        'cold_fraction':.25,'cold_sampling_probability':.5,'days':list(days),
        'cold_days':[days[i] for i in selected],'weights':weights.tolist(),
        'uniform_temperature_mean':float(temperature.mean()),
        'weighted_temperature_mean':float(weights@temperature),
        'validation_used_for_weight_fit':False,'test_evaluated':False}


class ColdTrainingEnv(FrozenQuantileEnv):
    def __init__(self,*args,split='train',**kwargs):
        if split!='train':raise ValueError('The sampler is training-only')
        super().__init__(*args,split=split,**kwargs)
        self.day_weights,self.sampling_info=cold_day_weights(self.table,self.days)
