"""Resolve e56 controller source locations while preserving all original guards."""
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import replay_correction_portable as historical

ROOT = Path(__file__).resolve().parents[2]
E56 = ROOT / 'Lee/experiments/e56'
sys.path.insert(0, str(E56))
import rate_io


@contextmanager
def portable_sources(records):
    def verify(mapping, root):
        historical.portable_verify(mapping, root, records)
    with patch.object(historical.policy_io, 'verify_sources', verify), \
            patch.object(rate_io, 'verify_sources', verify):
        yield


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--operation', choices=('compare', 'audit'), default='compare')
    known, remaining = parser.parse_known_args()
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument('--baseline', type=Path, required=True)
    output.add_argument('--out', type=Path, required=True)
    target, _ = output.parse_known_args(remaining)
    provenance = (target.out / 'portable_replay_provenance.json' if known.operation == 'compare'
                  else target.out.with_name(target.out.stem + '_portable_provenance.json'))
    if provenance.exists():
        raise FileExistsError(provenance)
    filename = 'compare_rates.py' if known.operation == 'compare' else 'audit_rates.py'
    spec = importlib.util.spec_from_file_location('e56_portable_operation', E56 / filename)
    operation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operation)
    records = []
    reference = json.loads((target.baseline.parent.parent / 'manifest.json').read_text())
    historical.portable_verify(reference['source_hashes'], ROOT, records)
    with portable_sources(records), patch.object(sys, 'argv', [str(E56 / filename), *remaining]):
        operation.main()
    unique = {json.dumps(row, sort_keys=True): row for row in records}
    result = {'operation': known.operation, 'source_location_adapter_only': True,
              'original_runtime_files_modified': False, 'source_hash_checks_preserved': True,
              'verified_sources': list(unique.values()),
              'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'shared_path_adapter_sha256': hashlib.sha256(Path(historical.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'scope': 'Map historical locations to actual current repository files. Preserve original base, regressor, configuration, budget, freeze, source and replay checks. Held-data audits additionally require the excluded captures.'}
    provenance.write_text(json.dumps(result, indent=2))
    print('PORTABLE_RATE_REPLAY_PASS', len(unique), flush=True)


if __name__ == '__main__':
    main()
