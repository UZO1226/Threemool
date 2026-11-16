"""Independent causal feature, initialization and rollout checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from long_context import LongContextData,LongContextModel,WINDOWS
from persistent_context import PersistentContextModel
from joint_teacher import TeacherTransitions
from full_day import sequences
from latent import Data,precompute,simulate

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
raw=pd.read_pickle(a.data);ref=json.loads(a.reference_summary.read_text())
data=LongContextData(raw,ref['fit_days']);base_data=Data(raw,ref['fit_days'])
all_days=ref['fit_days']+ref['calibration_days']+raw['days']['val']
all_positions=np.array([data.positions[day][0] for day in all_days])
assert np.all(all_positions>0)
assert np.isfinite(data.past(all_positions,16)).all()
assert data.table.index.max()<pd.Timestamp(raw['days']['val'][-1])+pd.Timedelta(days=1)
days=raw['days']['val'][:2];positions=np.array([data.positions[d][0] for d in days])
context=data.past(positions,16)
probes=np.r_[1,16,96,288,672,positions];expected=[]
for position in probes:
    blocks=[]
    for window in WINDOWS:
        values=data.context[max(position-window,0):position]
        blocks.extend([values.mean(0),values.std(0),values[-1]-values[0],np.array([len(values)/window])])
    expected.append(np.concatenate(blocks))
np.testing.assert_allclose(data.past(probes,16)[:,304:],expected,rtol=1e-8,atol=1e-8)
np.testing.assert_array_equal(context[:,:304],base_data.past(positions,16))

# Change every future validation input, including the start row, while keeping
# all fitting data intact. The already available initial history must not change.
changed={**raw,'data':raw['data'].copy()}
cutoff=data.table.index[positions[0]]
numeric=changed['data'].select_dtypes(include=[np.number]).columns
mask=(changed['data'].index>=cutoff)&(changed['data'].index<=data.table.index[-1])
rng=np.random.default_rng(41)
changed['data'].loc[mask,numeric]=rng.normal(100,30,(int(mask.sum()),len(numeric)))
scrambled=LongContextData(changed,ref['fit_days'])
np.testing.assert_array_equal(context[:1],scrambled.past(positions[:1],16))

torch.manual_seed(0);baseline=PersistentContextModel(data.sd,data.mean,data.mean_air,data.hi)
baseline_rng=torch.random.get_rng_state().clone()
torch.manual_seed(0);model=LongContextModel(data.sd,data.mean,data.mean_air,data.hi)
torch.testing.assert_close(baseline_rng,torch.random.get_rng_state(),rtol=0,atol=0)
for key,value in baseline.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
s,c,e,air,target=sequences(data,days)
with torch.no_grad():
    torch.testing.assert_close(model(s,c,e,air),baseline(s,c[:,:304],e,air),rtol=0,atol=0)
aux=TeacherTransitions(data,days)
loss=aux(model,torch.arange(len(days)));loss.backward()
gradient=model.long_encoder[-1].weight.grad
assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max()>0
optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.step();model.eval()
with torch.no_grad():
    predictions=model(s,c,e,air)
    local=aux.predictions(model,torch.arange(len(days)))
    torch.testing.assert_close(predictions[:,0],local[:,0])
cache=precompute([model],data,days,16)
numpy_prediction=simulate(cache,np.zeros(5))[0,0]
np.testing.assert_allclose(numpy_prediction,predictions.detach().numpy(),rtol=2e-4,atol=2e-4)
delta=float(model.long_encoder(c[:,304:]).detach().abs().max());assert delta>0
result={'direct_slice_statistics_passed':True,'short_context_unchanged':True,
    'all_fit_cal_validation_initial_contexts_finite':True,'history_table_stops_at_validation_end':True,
    'future_validation_scrambling_passed':True,'initial_parent_predictions_exact':True,
    'common_parameters_exact':True,'common_rng_preserved':True,'new_head_gradient_max':float(gradient.abs().max()),
    'nonzero_long_offset_after_update':delta,'teacher_first_step_matches_recursive':True,
    'full_day_torch_numpy_passed':True,'test_evaluated':False}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
