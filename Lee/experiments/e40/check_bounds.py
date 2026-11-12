"""Independent small-data regression for nominal quantile monotonicity."""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as implementation

from monotone_bounds import preserve_bounds

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
rng=np.random.RandomState(2);x=rng.uniform(-2,2,(600,2))
y=x[:,0]+rng.normal(size=600)*(.3+abs(x[:,1]))
probe=np.c_[np.linspace(-2,2,1000),np.full(1000,.3)]
def fit():return HGB(loss='quantile',quantile=.9,monotonic_cst=[1,0],max_iter=100,min_samples_leaf=10,random_state=0).fit(x,y)
original=implementation._update_leaves_values;old=fit().predict(probe)
old_bad=np.maximum(-np.diff(old),0);assert old_bad.max()>1e-4
with preserve_bounds(reconstruct=False):naive=fit().predict(probe)
naive_bad=float(np.maximum(-np.diff(naive),0).max());assert naive_bad>1e-4
with preserve_bounds() as stats:new=fit().predict(probe)
assert implementation._update_leaves_values is original and stats['restored']
assert np.isfinite(new).all() and np.maximum(-np.diff(new),0).max()<1e-10
assert stats['leaves_clipped']>0
result={'original_largest_decline':float(old_bad.max()),'original_reversal_count':int((old_bad>1e-8).sum()),
    'leaf_field_only_clamp_largest_decline':naive_bad,
    'corrected_largest_decline':float(np.maximum(-np.diff(new),0).max()),'correction':stats,
    'package_function_restored':True,'test_evaluated':False}
with open(a.out,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
