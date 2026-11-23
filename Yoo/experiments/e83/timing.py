"""Align the e81 matched paths to the first recovery, keeping the full-day denominator."""
import argparse
import csv
from pathlib import Path

import numpy as np


def increments(z):
    state = z['states']
    return np.stack([z['air']*.25, z['raw']*.25, state[:,:,0]>4,
                     state[:,:,2:].max(2)>2, np.maximum(0,state[:,:,0]-4)*.25],axis=2)


def timing(source, out):
    with out.open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['seed','elapsed_min','active_pairs','air','raw_air','nh','do','excess'])
        for seed in range(5):
            with np.load(source/f'{seed}_recover.npz') as rec, np.load(source/f'{seed}_pulse.npz') as pulse:
                np.testing.assert_array_equal(rec['dates'],pulse['dates'])
                cut = rec['delta'] < 0
                diff = increments(rec) - increments(pulse)
            assert diff.shape == (200,95,5)
            np.testing.assert_allclose(diff[cut.cumsum(1)==0],0,atol=1e-8,rtol=0)
            aligned = np.zeros_like(diff)
            counts = np.zeros(95,int)
            for ep in np.flatnonzero(cut.any(1)):
                t = cut[ep].argmax()
                aligned[ep,:95-t] = diff[ep,t:]
                counts[:95-t] += 1
            curve = aligned.mean(0).cumsum(0)
            np.testing.assert_allclose(curve[-1],diff.sum(1).mean(0),atol=1e-8,rtol=0)
            for k,values in enumerate(curve):
                writer.writerow([seed,(k+1)*15,counts[k],*values])
            cross = np.flatnonzero(curve[:,0]>1e-8)
            print(seed,'first positive minute',int(15*(cross[0]+1)) if len(cross) else None,
                  'end',curve[-1].round(3))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    timing(a.source,a.out)
