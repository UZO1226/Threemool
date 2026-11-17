"""Read-only verification of self-improvement archives against current GitHub assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'gh', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--require-fresh', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / 'Threemool'
    indexes = [root / f'Lee/experiments/{trial}/artifacts.json'
               for trial in ('e63', 'e65', 'e66', 'e67', 'e68', 'e69', 'e70', 'e71', 'e72')]
    indexes.append(root / 'Lee/experiments/e68/held_artifacts.json')
    fresh = root / 'Lee/experiments/e71/fresh_artifacts.json'
    if fresh.exists():
        indexes.append(fresh)
    else:
        assert not args.require_fresh, 'Fresh candidate delivery is missing'
    releases, rows = {}, []
    for path in indexes:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for key in ('source_data_included', 'residual_banks_included', 'pit_banks_included', 'captured_training_arrays_included'):
            assert manifest[key] is False
        remote, archive = manifest['remote'], manifest['archive']
        release_id = remote['release_id']
        if release_id not in releases:
            command = [str(args.gh.resolve()), 'api', f'repos/UZO1226/Threemool/releases/{release_id}']
            releases[release_id] = json.loads(subprocess.check_output(command, text=True, encoding='utf-8'))
        release = releases[release_id]
        assert release['draft'] and release['tag_name'] == remote['tag']
        asset = next(item for item in release['assets'] if item['id'] == remote['asset_id'])
        assert asset['name'] == archive['name'] and asset['size'] == archive['bytes']
        assert asset['digest'] == 'sha256:' + archive['sha256'] and asset['state'] == 'uploaded'
        local = workspace / 'results/artifacts' / archive['name']
        local_verified = False
        if local.exists():
            assert local.stat().st_size == archive['bytes'] and sha(local) == archive['sha256']
            with zipfile.ZipFile(local) as zipped:
                assert zipped.testzip() is None
                internal = json.loads(zipped.read('artifact_manifest.json'))
                assert internal['files'] == manifest['files']
                assert set(zipped.namelist()) == set(manifest['files']) | {'artifact_manifest.json'}
                for name, record in manifest['files'].items():
                    assert '/transitions/' not in name and 'pit_bank' not in name and 'residual_bank' not in name
                    content = zipped.read(name)
                    assert len(content) == record['bytes'] and hashlib.sha256(content).hexdigest() == record['sha256']
            local_verified = True
        rows.append({'index': path.relative_to(root).as_posix(), 'index_sha256': sha(path),
            'release_id': release_id, 'asset_id': asset['id'], 'archive': archive,
            'remote_size_digest_and_draft_verified': True, 'local_zip_entries_verified': local_verified,
            'file_count': len(manifest['files'])})
    result = {'verified': True, 'fresh_candidate_delivered': fresh.exists(), 'artifacts': rows,
        'new_uploads': 0, 'verifier_sha256': sha(Path(__file__)),
        'scope': 'Current remote asset identity, draft, size and SHA256 plus every locally available ZIP entry. Does not infer scientific success or goal completion.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('CURRENT_DELIVERY_VERIFIED', len(rows), 'fresh_candidate_delivered', fresh.exists(), flush=True)


if __name__ == '__main__':
    main()
