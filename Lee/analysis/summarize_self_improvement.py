"""Keep the e63 comparison block distinct from original-FCSRL trial tables."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    certificate_path = ROOT / 'Lee/experiments/e64/baseline_authentication.json'
    certificate = json.loads(certificate_path.read_text())
    assert certificate['passed'] and certificate['policy_seeds'] == list(range(5))
    inputs = {str(certificate_path.relative_to(ROOT)): sha(certificate_path)}
    indices = np.random.default_rng(2026).integers(0, 5, size=(10000, 5))
    completed, pending = [], []
    for trial in ('e65', 'e67', 'e69', 'e71'):
        folder = ROOT / f'Lee/experiments/{trial}'
        path = folder / 'comparison.json'
        if not path.exists():
            pending.append(trial)
            continue
        report = json.loads(path.read_text())
        assert report['baseline_method'] == 'e63_matched_risk_only_h4'
        assert report['policy_seeds'] == list(range(5)) and not report['test_evaluated']
        inputs[str(path.relative_to(ROOT))] = sha(path)
        for seed in range(5):
            audit_path = folder / f'audit_seed_{seed}.json'
            audit = json.loads(audit_path.read_text())
            assert audit['passed'] and audit['policy_seed'] == seed and not audit['test_evaluated']
            assert set(audit['replays']) == {'native', 'matched'}
            for replay in audit['replays'].values():
                assert replay['action_rows'] == 152000 and replay['independent_sklearn_actions_exact']
                assert replay['trajectory_replay_exact'] and replay['executed_air_accounting_exact']
            inputs[str(audit_path.relative_to(ROOT))] = sha(audit_path)
        matched = report['metrics']['candidate_matched']
        for mode in ('candidate_native', 'candidate_matched'):
            for metric, row in report['metrics'][mode].items():
                baseline = np.array(row['baseline_values'])
                candidate = np.array(row['candidate_values'])
                expected = [record['primary_reference_metrics'][metric] for record in certificate['records']]
                np.testing.assert_allclose(baseline, expected, atol=1e-12, rtol=0)
                delta = candidate - baseline
                interval = np.quantile(delta[indices].mean(1), [.025, .975])
                np.testing.assert_allclose(delta, row['paired_difference'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(interval, row['paired_seed_bootstrap95_ci'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(candidate.mean(), row['candidate_mean'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(candidate.std(ddof=1), row['candidate_sd'], atol=1e-12, rtol=0)
        target, actual = np.array(matched['air_m3']['baseline_values']), np.array(matched['air_m3']['candidate_values'])
        checks = {'all_five_seeds_air_matched': bool(np.all(np.abs(actual - target) <= target * .01)),
            'cost_difference_interval_below_zero': matched['cost']['paired_seed_bootstrap95_ci'][1] < 0,
            'nh4_difference_interval_below_zero': matched['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1] < 0,
            'do_difference_interval_not_above_zero': matched['do_viol_pct']['paired_seed_bootstrap95_ci'][1] <= 0}
        assert checks == report['checks']
        assert all(checks.values()) == report['meets_prespecified_simulation_criterion']
        completed.append({'trial': trial, 'checks': checks, 'passed': all(checks.values()),
                          'native': report['metrics']['candidate_native'], 'matched': matched})
    env_path = ROOT / 'Lee/experiments/e66/audit.json'
    env = json.loads(env_path.read_text())
    assert env['independent_donors_and_paths_exact'] and env['independent_full_day_passed']
    assert env['adoption']['passed'] == all(env['adoption']['checks'].values())
    assert not any(env['adoption']['checks'].values())
    inputs[str(env_path.relative_to(ROOT))] = sha(env_path)
    result = {'primary_reference': 'Authenticated matched e63, not original FCSRL or e63 native.',
        'completed_original_seed_block': completed, 'pending_committed_complete_reports': pending,
        'environment_e66_adoption': env['adoption'], 'source_sha256': inputs,
        'generator_sha256': sha(Path(__file__)), 'test_evaluated': False,
        'scope': 'Recompute committed aggregate arithmetic and require all independent seed audits; no new simulator query or policy gain.'}
    text = ['# Self-improvement trials versus e63', '',
        'This block uses authenticated matched e63 as its primary comparator.',
        'The earlier DARROW_AGENT_TRIALS.md table retains original FCSRL as its comparator.',
        'Keep the two blocks separate. Five original policy seeds0 through4 are used here.', '',
        '| Trial | Matched cost | NH4 percent | DO percent | Cost difference95% interval | NH4 interval, pp | DO interval, pp | Joint result |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def interval(row):
        return '[' + ','.join(f'{value:.6f}' for value in row['paired_seed_bootstrap95_ci']) + ']'
    for item in completed:
        m = item['matched']
        text.append(f"| [{item['trial']}](../experiments/{item['trial']}/results.md) | {m['cost']['candidate_mean']:.6f} | "
            f"{m['nh4_viol_pct']['candidate_mean']:.6f} | {m['do_viol_pct']['candidate_mean']:.6f} | "
            f"{interval(m['cost'])} | {interval(m['nh4_viol_pct'])} | {interval(m['do_viol_pct'])} | {'Pass' if item['passed'] else 'Fail'} |")
    text.extend(['', 'Differences are candidate minus e63. Cost sums the two violation indicators over95 steps,',
        'not money or electricity. Air is actual m3 per23.75-hour episode. All air matches, cost and',
        'NH4 upper endpoints below zero, and DO upper endpoint at most zero are required.',
        'Intervals use10000 paired policy-seed resamples, RNG2026. All native values remain in the JSON.', '',
        'Pending complete committed reports: ' + (', '.join(pending) if pending else 'none') + '.',
        'Pending does not prove a process is live or a partial outcome is a completed trial.', '',
        'Environment e66 independently reconstructs but fails all seven adoption conditions.',
        'It is a separate forecast experiment and does not replace e42 or enter the policy table.', '',
        'e71 passes this original seed block. Fresh seeds5 through9 remain a separate required',
        'confirmation block. Passing this table alone does not complete the active goal.',
        'Fresh baselines are authenticated in e68. No fresh candidate outcome is included here.',
        'Reused validation and multiple hypotheses remain exploratory; this is not independent',
        'observational confirmation or physical plant safety. Final test41 is preserved.', '',
        'Regenerate with summarize_self_improvement.py using committed compact reports only.', ''])
    (OUT / 'DARROW_SELF_IMPROVEMENT_TRIALS.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (OUT / 'DARROW_SELF_IMPROVEMENT_TRIALS.md').write_text('\n'.join(text))
    print('SELF_COMPARISON_RECOMPUTED', len(completed), 'complete;', pending, 'pending', flush=True)


if __name__ == '__main__':
    main()
