"""Respect existing grower bounds during a scoped quantile-leaf line search."""
from contextlib import contextmanager
import hashlib
import inspect
from threading import Lock

import numpy as np
import sklearn
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as implementation

EXPECTED_SOURCE='cf60fddf8e2fc786c2c548c91016a06f8fd0ece38182aac46372e77c5d5cbf43'


def inherited_leaf_bounds(grower):
    bounds={};stack=[(grower.root,-np.inf,np.inf)]
    while stack:
        node,lower,upper=stack.pop()
        if node.is_leaf:
            bounds[id(node)]=(lower,upper);continue
        direction=grower.monotonic_cst[node.split_info.feature_idx] if grower.with_monotonic_cst else 0
        if direction==0:
            left=(lower,upper);right=(lower,upper)
        else:
            mid=float(np.clip((node.split_info.value_left+node.split_info.value_right)/2,lower,upper))
            if direction>0:left=(lower,mid);right=(mid,upper)
            else:left=(mid,upper);right=(lower,mid)
        stack.append((node.left_child,*left));stack.append((node.right_child,*right))
    return bounds


@contextmanager
def preserve_bounds(reconstruct=True):
    assert sklearn.__version__=='1.9.1'
    original=implementation._update_leaves_values
    digest=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    assert digest==EXPECTED_SOURCE
    stats={'sklearn_version':sklearn.__version__,'original_source_sha256':digest,
        'leaves_clipped':0,'leaves_checked':0,'maximum_correction':0.,'restored':False,
        'reconstructed_bounds':reconstruct,'missing_bound_leaves':0}
    lock=Lock()
    def bounded(loss,grower,y_true,raw_prediction,sample_weight):
        original(loss,grower,y_true,raw_prediction,sample_weight)
        clipped=0;maximum=0.;missing=0
        limits=inherited_leaf_bounds(grower) if reconstruct else None
        for leaf in grower.finalized_leaves:
            before=leaf.value
            lo,hi=(leaf.children_lower_bound,leaf.children_upper_bound) if limits is None else limits[id(leaf)]
            if lo>leaf.children_lower_bound or hi<leaf.children_upper_bound:missing+=1
            lower=grower.shrinkage*lo
            upper=grower.shrinkage*hi
            leaf.value=float(np.clip(before,lower,upper))
            difference=abs(leaf.value-before)
            if difference>1e-14:clipped+=1;maximum=max(maximum,difference)
        with lock:
            stats['leaves_clipped']+=clipped;stats['leaves_checked']+=len(grower.finalized_leaves)
            stats['missing_bound_leaves']+=missing
            stats['maximum_correction']=max(stats['maximum_correction'],maximum)
    implementation._update_leaves_values=bounded
    try:yield stats
    finally:
        implementation._update_leaves_values=original
        stats['restored']=implementation._update_leaves_values is original

