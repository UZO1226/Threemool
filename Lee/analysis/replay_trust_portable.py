"""Resolve historical source locations for e63 without weakening replay guards."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import replay_rate_portable as inherited

ROOT = Path(__file__).resolve().parents[2]
E63 = ROOT / 'Lee/experiments/e63'
sys.path.insert(0, str(E63))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--operation', choices=('compare', 'audit'), default='audit')
    known, remaining = parser.parse_known_args()
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument('--baseline', type=Path, required=True)
    paths.add_argument('--out', type=Path, required=True)
    target, _ = paths.parse_known_args(remaining)
    provenance = (target.out / 'portable_replay_provenance.json' if known.operation == 'compare'
                  else target.out.with_name(target.out.stem + '_portable_provenance.json'))
    if provenance.exists():
        raise FileExistsError(provenance)
    filename = 'compare_trust.py' if known.operation == 'compare' else 'audit_trust.py'
    spec = importlib.util.spec_from_file_location('e63_portable_operation', E63 / filename)
    operation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operation)
    records = []
    reference = json.loads((target.baseline.parent.parent / 'manifest.json').read_text())
    inherited.historical.portable_verify(reference['source_hashes'], ROOT, records)
    with inherited.portable_sources(records), patch.object(sys, 'argv', [str(E63 / filename), *remaining]):
        operation.main()
    unique = {json.dumps(record, sort_keys=True): record for record in records}
    result = {'operation': known.operation, 'source_location_adapter_only': True,
              'original_runtime_files_modified': False, 'source_hash_checks_preserved': True,
              'verified_sources': list(unique.values()),
              'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'inherited_adapter_sha256': hashlib.sha256(Path(inherited.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'scope': 'Relocate source paths only; retain original base/rate/config/freeze/source/budget and action/trajectory checks.'}
    provenance.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('PORTABLE_TRUST_REPLAY_PASS', len(unique), flush=True)


if __name__ == '__main__':
    main()
