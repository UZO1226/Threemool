"""Compare logged action neutrality with actual native closed-loop air."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    results = args.workspace.resolve() / 'results'
    certificate_path = ROOT / 'Lee/experiments/e64/baseline_authentication.json'
    certificate = json.loads(certificate_path.read_text())
    preflight_path = results / 'e69_projection_fit/preflight.json'
    preflight = json.loads(preflight_path.read_text())
    manifest_path = results / 'e69_study/study_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert certificate['passed'] and preflight['passed']
    assert preflight['baseline_certificate_sha256'] == sha(certificate_path)
    assert preflight['policy_seeds'] == certificate['policy_seeds'] == list(range(5))
    for path, expected in manifest['source_sha256'].items():
        assert sha(Path(path)) == expected
    sources = {str(path): sha(path) for path in (certificate_path, preflight_path, manifest_path)}
    factor = 9000 * .25 * 95 / 2
    rows = []
    for seed in range(5):
        fit = next(row for row in preflight['records'] if row['policy_seed'] == seed)
        reference = next(row for row in certificate['records'] if row['policy_seed'] == seed)
        assert fit['base_checkpoint_hashes'] == reference['base_checkpoint_hashes']
        assert fit['rate_model_sha256'] == reference['rate_model_sha256']
        native_path = results / f'e69_study/comparisons/seed_{seed}/candidate_native.json'
        native = json.loads(native_path.read_text())['summary']
        sources[str(native_path)] = sha(native_path)
        actual_delta = native['air_m3'] - reference['e63_native_metrics']['air_m3']
        fit_delta = fit['calibration']['achieved_mean_action'] - fit['calibration']['target_mean_action']
        held_delta = fit['held_candidate_mean_action'] - fit['held_e63_native_mean_action']
        audit_path = native_path.parent / 'action_audit.json'
        audited = False
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            assert audit['passed'] and audit['policy_seed'] == seed
            audited = audit['replays']['native']['trajectory_replay_exact']
            sources[str(audit_path)] = sha(audit_path)
        rows.append({'policy_seed': seed, 'fit_mean_action_difference': fit_delta,
            'held_mean_action_difference': held_delta,
            'fit_static_equivalent_air_difference_m3': factor * fit_delta,
            'held_static_equivalent_air_difference_m3': factor * held_delta,
            'e63_native_air_m3': reference['e63_native_metrics']['air_m3'],
            'candidate_native_air_m3': native['air_m3'], 'actual_native_air_difference_m3': actual_delta,
            'actual_native_air_relative_difference': actual_delta / reference['e63_native_metrics']['air_m3'],
            'full_native_sklearn_trajectory_audit_available': bool(audited)})
    result = {'policy_seeds': list(range(5)), 'rows': rows,
        'mean_actual_native_air_difference_m3': float(np.mean([row['actual_native_air_difference_m3'] for row in rows])),
        'all_native_audits_available': all(row['full_native_sklearn_trajectory_audit_available'] for row in rows),
        'source_sha256': sources, 'generator_sha256': sha(Path(__file__)),
        'test_evaluated': False, 'new_environment_queries': 0, 'candidate_selection': False,
        'scope': 'Native air mechanism diagnostic, not a complete matched-air result or success claim.',
        'interpretation': 'Static equivalent volumes rescale mean actions on logged states; they are not executed closed-loop volumes. Native drift can reflect occupancy and date shift, but does not identify a cause.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'rows': rows, 'mean_actual_delta': result['mean_actual_native_air_difference_m3']}), flush=True)


if __name__ == '__main__':
    main()
