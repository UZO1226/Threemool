"""One-to-one onset matching without counting carried-in events as new."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def onset_counts(actual,predicted,initial,tolerance):
    actual=np.asarray(actual,bool);predicted=np.asarray(predicted,bool)
    true=np.flatnonzero(actual&~np.r_[bool(initial),actual[:-1]])
    forecast=np.flatnonzero(predicted&~np.r_[bool(initial),predicted[:-1]])
    if not len(true) or not len(forecast):return np.array([len(true),len(forecast),0,0.])
    distance=np.abs(true[:,None]-forecast[None,:])
    rows,columns=linear_sum_assignment(np.where(distance<=tolerance,distance,1e6))
    valid=distance[rows,columns]<=tolerance
    return np.array([len(true),len(forecast),int(valid.sum()),float(distance[rows[valid],columns[valid]].sum())])
