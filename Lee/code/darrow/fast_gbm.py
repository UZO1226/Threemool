"""Optional finite-input numeric HGB inference, checked against sklearn first."""
import numpy as np
from numba import njit


@njit(cache=True)
def predict(x, members, offsets, roots, feature, threshold, left, right, value, leaf, baseline):
    out=np.empty((len(x),5))
    for i in range(len(x)):
        for j in range(5):
            model=members[i]*5+j
            total=baseline[model]
            for t in range(offsets[model],offsets[model+1]):
                root=roots[t]
                node=root
                while not leaf[node]:
                    if x[i,feature[node]]<=threshold[node]:
                        node=root+left[node]
                    else:
                        node=root+right[node]
                total+=value[node]
            out[i,j]=total
    return out


class FastPredictor:
    def __init__(self, models):
        nodes,roots,offsets,baseline=[],[],[0],[]
        self.direct=[]
        coefficients=[]
        temperature_coefficients=[]
        temperature_centers=[]
        n=0
        for row in models:
            for wrapped in row:
                direct=hasattr(wrapped,'target')
                model=wrapped.model if direct else wrapped
                if model.is_categorical_ is not None and np.any(model.is_categorical_):
                    raise ValueError('Fast inference only supports numeric trees')
                baseline.append(float(model._baseline_prediction[0,0]))
                self.direct.append(direct)
                coefficients.append(float(getattr(wrapped,'state_coefficient',0)))
                temperature_coefficients.append(float(getattr(wrapped,'temperature_coefficient',0)))
                temperature_centers.append(float(getattr(wrapped,'temperature_center',0)))
                for predictors in model._predictors:
                    if len(predictors)!=1:
                        raise ValueError('Expected a single regressor tree per iteration')
                    tree=predictors[0].nodes
                    if tree['is_categorical'].any():
                        raise ValueError('Categorical splits unsupported')
                    roots.append(n)
                    nodes.append(tree)
                    n+=len(tree)
                offsets.append(len(roots))
        all_nodes=np.concatenate(nodes)
        self.args=[np.asarray(offsets),np.asarray(roots),*[np.ascontiguousarray(all_nodes[k]) for k in
            ('feature_idx','num_threshold','left','right','value','is_leaf')],np.asarray(baseline)]
        self.direct=np.asarray(self.direct).reshape(len(models),5)
        self.coefficients=np.asarray(coefficients).reshape(len(models),5)
        self.temperature_coefficients=np.asarray(temperature_coefficients).reshape(len(models),5)
        self.temperature_centers=np.asarray(temperature_centers).reshape(len(models),5)

    def __call__(self,x,members):
        if not np.isfinite(x).all():
            raise ValueError('Fast inference requires finite input')
        out=predict(np.asarray(x,dtype=np.float64),members,*self.args)
        return (out+self.coefficients[members]*x[:,:5]
                +self.temperature_coefficients[members]*(x[:,6:7]-self.temperature_centers[members])
                -np.where(self.direct[members],x[:,:5],0))
