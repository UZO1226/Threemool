"""Secondary e74/e73 contrast from committed, independently audited reports."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    reports, sources = {}, {}
    for trial in ('e73', 'e74'):
        folder = ROOT / f'Lee/experiments/{trial}'
        path = folder / 'comparison.json'
        report = json.loads(path.read_text())
        assert report['policy_seeds'] == list(range(5)) and not report['test_evaluated']
        assert report['baseline_method'] == 'e63_matched_risk_only_h4'
        assert not report['meets_prespecified_simulation_criterion']
        reports[trial] = report
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in range(5):
            path = folder / f'audit_seed_{seed}.json'
            audit = json.loads(path.read_text())
            assert audit['passed'] and audit['policy_seed'] == seed and not audit['test_evaluated']
            assert set(audit['replays']) == {'native', 'matched'}
            assert all(row['action_rows'] == 152000 and row['trajectory_replay_exact'] and
                       row['independent_sklearn_actions_exact'] and row['wrapper_component_guard_passed']
                       for row in audit['replays'].values())
            sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert reports['e73']['freeze_canonical_sha256'] == reports['e74']['freeze_canonical_sha256']
    ix = np.random.default_rng(2026).integers(0, 5, (10000, 5))
    modes = {}
    for mode in ('candidate_native', 'candidate_matched'):
        rows = {}
        for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct'):
            a, b = [np.array(reports[trial]['metrics'][mode][key]['candidate_values']) for trial in ('e73', 'e74')]
            delta = b - a
            rows[key] = {'e73_mean': float(a.mean()), 'e74_mean': float(b.mean()),
                'paired_difference': delta.tolist(), 'mean_difference': float(delta.mean()),
                'paired_policy_seed95_ci': np.quantile(delta[ix].mean(1), [.025, .975]).tolist()}
        modes[mode] = rows
    result = {'primary_reference_unchanged': 'Authenticated matched e63', 'secondary_reference': 'e73',
        'candidate': 'e74', 'policy_seeds': list(range(5)), 'metrics': modes,
        'both_primary_joint_criteria_fail': True, 'test_evaluated': False, 'new_environment_queries': 0,
        'source_sha256': sources, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Secondary same-policy-seed mechanism contrast; both matched policies target the same e63 air. Native is native versus native. Not a rescue of failed DO, isolated causal ablation, physical safety or unseen validation.'}
    path = ROOT / 'Lee/experiments/e74/mechanism_contrast.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(modes), flush=True)


if __name__ == '__main__':
    main()
