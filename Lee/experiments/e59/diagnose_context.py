"""Fixed context and observed-state diagnosis after the completed e58 failure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e58'))
from upper_tail import UpperDynamics, interpolate, q
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics


def distribution(values):
    return {'min': float(values.min()), 'q25': float(np.quantile(values, .25)),
            'median': float(np.median(values)), 'q75': float(np.quantile(values, .75)),
            'max': float(values.max()), 'mean': float(values.mean())}


def context(x):
    return {'nh4': x[:, 0], 'max_do': x[:, 2:5].max(1), 'flow': x[:, 5],
            'temperature': x[:, 6], 'air_m3_per_h': x[:, 14:19].sum(1) * 1000}


def authenticate(directory, data_digest):
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['data_sha256'] == data_digest and summary['test_evaluated'] is False
    assert summary['selected'] == 'pit_b95_a0.25'
    for name, expected in summary['model_hashes'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e58', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    sources = {name: authenticate(directory, digest) for name, directory in [('e51', args.e51), ('e58', args.e58)]}
    assert sources['e51']['fit_days'] == sources['e58']['fit_days']
    assert sources['e51']['calibration_days'] == sources['e58']['calibration_days']
    records = {'e51': ROOT / 'Lee/experiments/e51/event_reference_audit.json',
               'e58': args.e58 / 'event_audit.json'}
    stored = {}
    for name, path in records.items():
        report = json.loads(path.read_text())
        assert report['test_evaluated'] is False
        assert report['selected'] == sources[name]['selected']
        stored[name] = report['metrics']['candidate']
        for label in ('nh4', 'do'):
            np.testing.assert_allclose(stored[name][label]['brier'],
                sources[name]['validation'][sources[name]['selected']][label + '_brier'], rtol=1e-12)
    raw = pd.read_pickle(args.data)
    splits = {'fit': sources['e58']['fit_days'], 'calibration': sources['e58']['calibration_days'],
              'validation': raw['days']['val']}
    assert list(map(len, splits.values())) == [158, 30, 40]
    for name, days in splits.items():
        assert set(days).isdisjoint(raw['days']['test'])
        assert all(set(days).isdisjoint(other) for key, other in splits.items() if key != name)
    result = {'scope': 'Post-outcome descriptive diagnosis, reused validation; observed-state reset is not a deployable-world gain.',
              'data_sha256': digest, 'model_refit': False, 'selection_changed': False,
              'test_evaluated': False, 'policy_seeds': [], 'model_members': 5, 'draws': 8,
              'calibration_warning': 'Same dates used for PIT estimation; not held-out evidence.',
              'context': {}, 'observed_state': {}, 'stored_recursive_validation': {}}
    arrays = {}
    for name, days in splits.items():
        x, delta = q.build_arrays(raw['data'], days)
        truth = q.arrays(raw['data'], days)[0]
        np.testing.assert_allclose(x[:, :5], truth[:, :-1].reshape(-1, 5), rtol=1e-12)
        np.testing.assert_allclose(x[:, :5] + delta, truth[:, 1:].reshape(-1, 5), rtol=1e-12, atol=1e-12)
        arrays[name] = (x, truth)
        target = truth[:, 1:]
        result['context'][name] = {'days': len(days), 'rows': len(x),
            'features': {key: distribution(value) for key, value in context(x).items()},
            'actual_nh4_frequency': float((target[..., 0] > 4).mean()),
            'actual_do_frequency': float((target[..., 2:].max(-1) > 2).mean())}
    result['validation_outside_context_ranges'] = {}
    for name in ('fit', 'calibration'):
        result['validation_outside_context_ranges'][name] = {
            key: float(((values < context(arrays[name][0])[key].min()) |
                        (values > context(arrays[name][0])[key].max())).mean())
            for key, values in context(arrays['validation'][0]).items()}
    backbone = q.QuantileDynamics(args.e51)
    candidate = UpperDynamics(q.QuantileDynamics(args.e58), q.GBMDynamics(args.e58 / 'upper_do_0.99.pkl'))
    for split in ('calibration', 'validation'):
        x, truth = arrays[split]
        days = splits[split]
        members = np.repeat(np.arange(5), len(x))
        tiled = np.tile(x, (5, 1))
        central = backbone.quantiles(tiled, members)
        extended = candidate.quantiles(tiled, members)
        np.testing.assert_array_equal(central, extended[:5])
        result['observed_state'][split] = {}
        for name, directory, values in [('e51', args.e51, central), ('e58', args.e58, extended)]:
            noise = q.paths(np.load(directory / 'pit_bank.npz')['pit'], days, 95).reshape(-1, 5)
            expanded = np.repeat(values.reshape(len(values), 5, 1, len(days), 95, 5), 8, axis=2).reshape(len(values), -1, 5)
            current = np.tile(x[:, :5], (5 * 8, 1))
            sampled = interpolate(expanded, noise) if name == 'e58' else q.interpolate(expanded, noise)
            prediction = np.clip(.25 * current + .75 * sampled, 0, backbone.hi).reshape(5, 8, len(days), 95, 5)
            metrics, _ = event_metrics(prediction, truth)
            result['observed_state'][split][name] = {'event_metrics': metrics,
                'nh4_frequency': float((prediction[..., 0] > 4).mean()),
                'do_frequency': float((prediction[..., 2:].max(-1) > 2).mean())}
            print('SCORED', split, name, flush=True)
    result['stored_recursive_validation'] = stored
    result['input_target_alignment_checked'] = True
    result['backbone_quantiles_exact'] = True
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
