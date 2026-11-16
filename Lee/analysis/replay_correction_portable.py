"""Resolve historical e55 source locations without relaxing their exact hashes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
E55 = ROOT / 'Lee/experiments/e55'
sys.path.insert(0, str(E55))
import policy_io


def resolve_source(name, root):
    normalized = str(name).replace('\\', '/')
    marker = '/Threemool/'
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        resolved = (ROOT / relative).resolve()
        resolved.relative_to(ROOT.resolve())
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    path = Path(name)
    if not path.is_absolute():
        path = root / path
    path.resolve().relative_to(ROOT.resolve())
    if path.exists():
        return path
    raise FileNotFoundError(path)


def portable_verify(mapping, root, records=None):
    for name, expected in mapping.items():
        path = resolve_source(name, root)
        assert expected in policy_io.freeze_identities(path), f'Source changed: {path}'
        if records is not None:
            records.append({'recorded_name': name, 'resolved_path': str(path.resolve()),
                            'expected_sha256': expected})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--operation', choices=('compare', 'action-audit'), default='compare')
    known, remaining = parser.parse_known_args()
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument('--out', type=Path, required=True)
    output.add_argument('--baseline', type=Path, required=True)
    target, _ = output.parse_known_args(remaining)
    provenance = (target.out / 'portable_replay_provenance.json' if known.operation == 'compare'
                  else target.out.with_name(target.out.stem + '_portable_provenance.json'))
    if provenance.exists():
        raise FileExistsError(provenance)
    filename = 'compare.py' if known.operation == 'compare' else 'audit_correction.py'
    spec = importlib.util.spec_from_file_location('e55_portable_operation', E55 / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    reference_manifest = json.loads((target.baseline.parent.parent / 'manifest.json').read_text())
    portable_verify(reference_manifest['source_hashes'], ROOT, records)
    def verifier(mapping, root):
        portable_verify(mapping, root, records)
    with patch.object(policy_io, 'verify_sources', verifier), \
            patch.object(sys, 'argv', [str(E55 / filename), *remaining]):
        module.main()
    unique = {json.dumps(row, sort_keys=True): row for row in records}
    result = {'operation': known.operation, 'source_location_adapter_only': True,
              'source_hash_checks_preserved': True, 'verified_sources': list(unique.values()),
              'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'original_runtime_files_modified': False,
              'scope': 'Historical source paths resolve to the actual current repository files, even if an older checkout still exists. All original checksum, checkpoint, configuration, budget, freeze and replay guards remain active.'}
    provenance.write_text(json.dumps(result, indent=2))
    print('PORTABLE_SOURCE_REPLAY_PASS', len(unique), flush=True)


if __name__ == '__main__':
    main()
