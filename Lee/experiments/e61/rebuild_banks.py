"""Rebuild excluded parent PIT banks from authenticated calibration inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime import q, load_model
from upper_tail import pit as upper_pit


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e58', 'preflight', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    preflight = json.loads(args.preflight.read_text())
    assert preflight['both_selected_banks_reconstructed_bitwise']
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == preflight['data_sha256']
    raw = pd.read_pickle(args.data)
    results = {}
    args.out.mkdir(parents=True, exist_ok=False)
    for parent, directory in [('e51', args.e51), ('e58', args.e58)]:
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['selected'] == 'pit_b95_a0.25' and summary['test_evaluated'] is False
        assert summary['data_sha256'] == preflight['data_sha256']
        for name, expected in summary['model_hashes'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        cal = summary['calibration_days']
        assert len(cal) == 30
        assert set(cal).isdisjoint(summary['fit_days'] + raw['days']['val'] + raw['days']['test'])
        x, delta = q.build_arrays(raw['data'], cal)
        current = np.tile(x[:, :5], (5, 1))
        target = np.tile(x[:, :5] + delta, (5, 1))
        model = load_model(directory, parent)
        values = model.quantiles(np.tile(x, (5, 1)), np.repeat(np.arange(5), len(x)))
        mixed = np.clip(.25 * current[None] + .75 * values, 0, model.hi)
        cdf = q.pit if parent == 'e51' else upper_pit
        bank = cdf(mixed, target, np.random.default_rng(3801)).reshape(5, len(x), 5)
        bank[..., 1] = .5
        digest = hashlib.sha256(bank.tobytes()).hexdigest()
        assert digest == preflight['bank_array_sha256'][parent]
        original = directory / 'pit_bank.npz'
        if original.exists():
            np.testing.assert_array_equal(bank, np.load(original)['pit'])
        destination = args.out / (parent + '_pit_bank.npz')
        np.savez_compressed(destination, pit=bank)
        np.testing.assert_array_equal(bank, np.load(destination)['pit'])
        results[parent] = {'array_sha256': digest, 'shape': list(bank.shape),
                           'existing_original_compared': original.exists(), 'output': destination.name}
    result = {'banks': results, 'model_refit': False, 'validation_evaluated': False,
              'test_evaluated': False, 'data_sha256': preflight['data_sha256'],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / 'rebuild_check.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
