"""Reconstruct the exploratory time profile for the selected interventions."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/jio/fcsrl_src/results')


def profile(root):
    records = json.loads((root / 'e93/main/summary.json').read_text())
    curves = []
    for row in records:
        seed = row['seed']
        with np.load(root / f'e93/main/gate{seed}.npz') as gate, np.load(root / f'e90/main/paths{seed}.npz') as src:
            np.testing.assert_array_equal(gate['keys'], src['keys'])
            np.testing.assert_array_equal(gate['supported'], src['supported'])
            use = gate['supported']
            cut = ~gate['skip'][use]
            day = gate['keys'][use, 0]
            sums = src['cumulative'][use]
            delta = (sums[:, 1, :, 0] - sums[:, 0, :, 0]) * cut[:, None]
            curve = np.mean([delta[day == d].mean(axis=0) for d in np.unique(day)], axis=0)
            np.testing.assert_allclose(curve[-1], row['groups']['supported']['delta']['gate'][0], atol=1e-7, rtol=0)
            curves.append(curve)
    return np.mean(curves, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    curve = profile(parser.parse_args().root)
    for step in [1, 4, 12, 15, 24]:
        print(f'{step * 15} min: {curve[step - 1]:+.2f} m³')
