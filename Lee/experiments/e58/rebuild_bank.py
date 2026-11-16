"""Rebuild the excluded selected PIT bank without fitting or validation access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_tail import UpperDynamics, q
from evaluate_upper import prepare


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary = json.loads((args.runs / 'summary.json').read_text())
    assert summary['test_evaluated'] is False
    assert summary['data_sha256'] == hashlib.sha256(args.data.read_bytes()).hexdigest()
    for name, digest in summary['model_hashes'].items():
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == digest
    raw = pd.read_pickle(args.data)
    days = summary['calibration_days']
    assert len(days) == 30
    assert set(days).isdisjoint(summary['fit_days'] + raw['days']['val'] + raw['days']['test'])
    dyn = UpperDynamics(q.QuantileDynamics(args.runs), q.GBMDynamics(args.runs / 'upper_do_0.99.pkl'))
    banks, _ = prepare(dyn, raw, days)
    alpha = float(summary['selected'].split('_')[2][1:])
    bank = banks[alpha]
    existing = args.runs / 'pit_bank.npz'
    if existing.exists():
        np.testing.assert_array_equal(bank, np.load(existing)['pit'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pit=bank)
    np.testing.assert_array_equal(bank, np.load(args.out)['pit'])
    print(json.dumps({'selected': summary['selected'], 'shape': list(bank.shape),
                      'array_sha256': hashlib.sha256(bank.tobytes()).hexdigest(),
                      'existing_bank_compared': existing.exists(), 'weight_fitting': False,
                      'validation_evaluated': False, 'test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()
