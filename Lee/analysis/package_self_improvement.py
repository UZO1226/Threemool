"""Archive only completed and independently audited self-improvement trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
from report import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('e65', 'e66'), required=True)
    for name in ('workspace', 'study', 'out', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.index.exists():
        raise FileExistsError('Refusing to overwrite archive or index')
    workspace = args.workspace.resolve()
    results = workspace / 'results'
    files = {}
    def add(path):
        relative = path.resolve().relative_to(workspace).as_posix()
        assert path.is_file() and path.suffix in ('.json', '.log', '.pkl', '.pt')
        assert '/transitions/' not in relative and 'pit_bank' not in relative
        files[relative] = path
    if args.kind == 'e65':
        status = json.loads((args.study / 'study_status.json').read_text())
        assert status['phase'] == 'complete' and not status['running'] and not status['pending']
        cert_path = ROOT / 'Lee/experiments/e64/baseline_authentication.json'
        cert = json.loads(cert_path.read_text())
        assert cert['passed']
        pairs = []
        for seed in range(5):
            folder = args.study / f'comparisons/seed_{seed}'
            pair = json.loads((folder / 'summary.json').read_text())
            audit = json.loads((folder / 'action_audit.json').read_text())
            assert pair['candidate_method'] == 'e65_nested_grid_h4'
            assert pair['policy_seed'] == audit['policy_seed'] == seed
            assert pair['training_steps'] == 401280 and not pair['pipeline_smoke_only']
            assert not pair['test_evaluated'] and not pair['smoke_only']
            assert pair['checkpoint_replay_verified'] and audit['passed']
            assert pair['baseline_certificate_sha256'] == sha(cert_path)
            assert pair['baseline_native'] == cert['records'][seed]['primary_reference_metrics']
            assert audit['comparison_summary_sha256'] == sha(folder / 'summary.json')
            assert set(audit['replays']) == {'native', 'matched'}
            for replay in audit['replays'].values():
                assert replay['action_rows'] == 152000
                assert replay['independent_sklearn_actions_exact'] and replay['trajectory_replay_exact']
                assert replay['executed_air_accounting_exact']
            assert sha(folder / 'controller_config.json') == pair['controller_config_sha256']
            base = results / f'e55_study/policies/new_ppolag_fcsrl/seed_{seed}'
            for name, expected in pair['base_checkpoint_hashes'].items():
                path = base / 'model_final' / name
                assert sha(path) == expected
                add(path)
            for name in ('effective_config.json', 'metrics.json', 'base_equivalence.json', 'correction/fit_summary.json'):
                add(base / name)
            model = results / f'e56_study/models/h4/seed_{seed}'
            assert sha(model / 'risk_models.pkl') == pair['correction_model_sha256']
            add(model / 'risk_models.pkl')
            add(model / 'fit_summary.json')
            pairs.append(pair)
        assert summarize(pairs) == json.loads((args.study / 'comparison.json').read_text())
        study_manifest = json.loads((args.study / 'study_manifest.json').read_text())
        for path, expected in study_manifest['source_sha256'].items():
            assert sha(Path(path)) == expected
        if 'previous_terminal_manifest_sha256' in study_manifest:
            previous = results / 'e65_study'
            assert sha(previous / 'study_manifest.json') == study_manifest['previous_terminal_manifest_sha256']
            for path in previous.rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.log'):
                    add(path)
        add(cert_path)
        for path in (results / 'e65_native_smoke').glob('*.json'):
            add(path)
        for path in (results / 'e65_preflight_v2').glob('*.json'):
            add(path)
    else:
        summary = json.loads((args.study / 'summary.json').read_text())
        audit = json.loads((args.study / 'audit.json').read_text())
        assert not summary['test_evaluated'] and not audit['test_evaluated']
        assert summary['primary'] == 'context32_block4'
        assert audit['independent_donors_and_paths_exact'] and audit['independent_full_day_passed']
        assert audit['summary_sha256'] == sha(args.study / 'summary.json')
        assert audit['adoption']['passed'] == all(audit['adoption']['checks'].values())
        add(results / 'e66_block_preflight.json')
    for path in args.study.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.log'):
            add(path)
    add(ROOT / 'Lee/experiments/e42/environment_freeze.json')
    files['metadata/FCSRL_LICENSE'] = ROOT / 'Lee/code/darrow/FCSRL_LICENSE'
    freeze = json.loads((ROOT / 'Lee/experiments/e42/environment_freeze.json').read_text())
    manifest = {'trial': args.kind, 'new_model_fitting': False, 'source_data_included': False,
                'residual_banks_included': False, 'pit_banks_included': False,
                'captured_training_arrays_included': False, 'test_evaluated': False,
                'frozen_environment_dependencies': freeze['frozen_files_sha256'],
                'scope': 'Audited aggregate outputs and provenance. e65 includes unchanged policy/rate weights; e66 reuses archived e40 weights. Frozen model/data dependencies require separate restoration.',
                'files': {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(files.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('artifact_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.out) as archive:
        assert archive.testzip() is None
        for name, entry in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256']
    manifest['archive'] = {'name': args.out.name, 'bytes': args.out.stat().st_size, 'sha256': sha(args.out)}
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(manifest['archive']), flush=True)


if __name__ == '__main__':
    main()
