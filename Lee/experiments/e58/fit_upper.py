"""Fit the prospectively fixed fifteen DO upper heads on training dates only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

from upper_tail import ROOT, UpperDynamics, q

sys.path.insert(0, str(ROOT / 'Lee/experiments/e40'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from monotone_bounds import preserve_bounds
from quantile_env import FrozenDynamics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'parent-freeze', 'preflight', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    preflight = json.loads(args.preflight.read_text())
    for key in ('all_four_zero_banks_bitwise_e51', 'complete_95_step_zero_rollout_bitwise'):
        assert preflight[key] is True
    assert preflight['data_sha256'] == digest(args.data)
    for name, expected in preflight['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    parent = FrozenDynamics(args.parent_freeze)
    reference = json.loads(args.reference_summary.read_text())
    raw = pd.read_pickle(args.data)
    fit = reference['fit_days']
    cal = reference['calibration_days']
    assert len(fit) == 158 and len(cal) == 30
    assert len(raw['days']['val']) == 40 and len(raw['days']['test']) == 41
    groups = [set(fit), set(cal), set(raw['days']['val']), set(raw['days']['test'])]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i))
    assert set(fit + cal) == set(raw['days']['train'])
    source = json.loads((parent.path / 'summary.json').read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for name, expected in source['model_hashes'].items():
        assert digest(parent.path / name) == expected
        shutil.copy2(parent.path / name, args.out / name)
        assert digest(args.out / name) == expected
    x, delta = q.build_arrays(raw['data'], fit)
    target = x[:, :5] + delta
    masked = x.copy()
    masked[:, 1] = 0
    constraint = np.zeros(x.shape[1], dtype=int)
    constraint[0] = -1
    constraint[2:5] = 1
    constraint[14:24] = 1
    subsets = []
    for member in range(5):
        selected = np.random.RandomState(member).choice(len(fit), int(.8 * len(fit)), replace=False)
        subsets.append((selected[:, None] * 95 + np.arange(95)).reshape(-1))

    def job(pair):
        member, state = pair
        model = HGB(loss='quantile', quantile=.99, max_iter=200, learning_rate=.05,
                    max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1,
                    monotonic_cst=constraint, early_stopping=False,
                    random_state=100 * member + state)
        index = subsets[member]
        model.fit(masked[index], target[index, state])
        assert len(model._predictors) == 200
        assert not any(np.any(tree[0].nodes['feature_idx'][tree[0].nodes['is_leaf'] == 0] == 1)
                       for tree in model._predictors)
        print('TRAINED', member, state, flush=True)
        return member, state, q.DirectDeltaRegressor(model, state)

    with preserve_bounds() as bounds:
        with ThreadPoolExecutor(max_workers=4) as pool:
            fitted = list(pool.map(job, [(m, j) for m in range(5) for j in (2, 3, 4)]))
    assert bounds['restored']
    # Non-DO placeholders reuse existing heads and are ignored by UpperDynamics.
    rows = [list(parent.model.models[-1].models[m]) for m in range(5)]
    for member, state, model in fitted:
        rows[member][state] = model
    with (args.out / 'upper_do_0.99.pkl').open('wb') as stream:
        pickle.dump({'models': rows, 'hi': parent.hi, 'bias': np.zeros(5)}, stream)
    upper = q.GBMDynamics(args.out / 'upper_do_0.99.pkl')
    dyn = UpperDynamics(q.QuantileDynamics(args.out), upper)
    checks = q.checks(dyn, x, strict_order=True)
    probe = x[np.linspace(0, len(x) - 1, 512, dtype=int)]
    members = np.arange(len(probe)) % 5
    values = dyn.quantiles(probe, members)
    np.testing.assert_array_equal(values[:5], parent.model.quantiles(probe, members))
    np.testing.assert_array_equal(values[-1, :, :2], values[-2, :, :2])
    assert np.all(np.diff(values, axis=0) >= 0)
    result = {
        'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                         'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'data_sha256': digest(args.data), 'preflight_sha256': digest(args.preflight),
        'parent_freeze_sha256': digest(args.parent_freeze),
        'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        'model_hashes': {p.name: digest(p) for p in args.out.glob('*.pkl')},
        'backbone_hashes_exact': source['model_hashes'], 'new_regressor_count': len(fitted),
        'fit_days': fit, 'calibration_days': cal, 'fit_rows': len(x),
        'subset_rows_per_member': [len(index) for index in subsets],
        'checks': checks, 'leaf_bound_enforcement': bounds,
        'central_backbone_predictions_exact': True, 'non_do_endpoint_unchanged': True,
        'validation_evaluated': False, 'test_evaluated': False,
        'status': 'fit_complete_pending_calibration_validation_and_independent_audit'}
    (args.out / 'fit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('FIT_COMPLETE', json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
