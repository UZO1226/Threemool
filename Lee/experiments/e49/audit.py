"""Reconstruct diagnostic predictions from saved classifiers and raw rows."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/code/darrow'))
from surrogate_dw import STATE, EXO, ZONES


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary = json.loads((args.runs / 'summary.json').read_text())
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == summary['data_sha256']
    checkpoint = args.runs / 'models.pkl'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary['model_sha256']
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    raw = pd.read_pickle(args.data)
    assert summary['test_evaluated'] is False and summary['policy_training'] is False
    assert set(summary['fit_days']).isdisjoint(raw['days']['test'])
    max_error = 0.
    for member in range(5):
        selected = np.random.RandomState(member).choice(158, int(.8 * 158), replace=False)
        assert summary['model_date_subsets'][str(member)] == [summary['fit_days'][i] for i in selected]
    counts = {}
    for split in ('calibration', 'validation'):
        days = summary[split + '_days']
        assert set(days).isdisjoint(summary['fit_days']) and set(days).isdisjoint(raw['days']['test'])
        targets, currents, inputs = [], [], []
        for day in days:
            group = raw['data'].loc[day]
            s, z = group[STATE].to_numpy(float), group[ZONES].to_numpy(float)
            assert len(s) == 96
            inputs.append(np.c_[s[:-1], group[EXO].to_numpy(float)[:-1],
                                z[:-1] / 1000, z[np.maximum(np.arange(95) - 1, 0)] / 1000])
            targets.append(np.c_[s[1:, 0] > 4, s[1:, 2:].max(-1) > 2])
            currents.append(np.c_[s[:-1, 0] > 4, s[:-1, 2:].max(-1) > 2])
        x, target, current = np.concatenate(inputs), np.stack(targets), np.stack(currents)
        predictions = np.load(args.runs / f'{split}_probabilities.npz')
        np.testing.assert_array_equal(predictions['persistence'], current)
        for name in ('full', 'no_state'):
            cols = saved['columns'][name]
            np.testing.assert_array_equal(cols, np.arange(24) if name == 'full' else np.arange(5, 24))
            values = []
            for member in saved['models'][name]:
                assert len(member) == 2
                for model in member:
                    assert model.n_iter_ == 200 and not model.early_stopping
                    assert model.class_weight is None and model.monotonic_cst is None
                values.append(np.stack([model.predict_proba(x[:, cols])[:, 1] for model in member], -1))
            probability = np.mean(values, axis=0).reshape(len(days), 95, 2)
            np.testing.assert_allclose(probability, predictions[name], rtol=0, atol=1e-14)
            max_error = max(max_error, float(np.abs(probability - predictions[name]).max()))
        for name in predictions.files:
            for j, label in enumerate(('nh4', 'do')):
                probability = predictions[name][..., j]
                for subset, mask in [('all_steps', np.ones(target.shape[:2], bool)),
                                     ('currently_nonviolating', ~current[..., j])]:
                    y, p = target[..., j][mask], probability[mask]
                    reported = summary['splits'][split][name][label][subset]
                    assert reported['count'] == len(y) and reported['positives'] == int(y.sum())
                    np.testing.assert_allclose(reported['brier'], np.mean((p - y) ** 2), rtol=1e-12)
                    np.testing.assert_allclose(reported['average_precision'], average_precision_score(y, p), rtol=1e-12)
                    np.testing.assert_allclose(reported['roc_auc'], roc_auc_score(y, p), rtol=1e-12)
        counts[split] = {'transitions': int(target.shape[0] * target.shape[1]),
                         'new_nh4': int((target[..., 0] & ~current[..., 0]).sum()),
                         'new_do': int((target[..., 1] & ~current[..., 1]).sum())}
    result = {'passed': True, 'test_evaluated': False, 'policy_training': False,
              'checkpoint_sha256_passed': True, 'data_sha256_passed': True,
              'fit_date_subsets_passed': True, 'direct_row_feature_reconstruction_passed': True,
              'classifier_probability_reconstruction_max_error': max_error,
              'all_and_new_onset_metrics_reconstructed': True, 'counts': counts}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
