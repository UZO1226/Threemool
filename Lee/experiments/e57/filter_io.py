"""Retain original policy/freeze/source guards for the fixed actuator extension."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e55'))
from policy_io import load_captured, load_reference, file_hashes
from quantile_env import FrozenDynamics, frozen_digest


def check_preflight(freeze):
    check = json.loads(Path(__file__).with_name('preflight.json').read_text())
    assert check['passed'] and not check['test_evaluated']
    assert check['freeze_canonical_sha256'] == frozen_digest(freeze)
    for name, expected in check['source_sha256'].items():
        assert frozen_digest(Path(__file__).with_name(name)) == expected


def load_base(run, freeze, baseline):
    check_preflight(freeze)
    return load_captured(run, freeze, baseline, False)
