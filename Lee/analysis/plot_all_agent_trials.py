"""Show every completed fixed-environment extension, retaining failed controls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIALS = [
    ('Split heads', 'ppolag_fcsrl_split', 'e43/split_comparison.json'),
    ('Capacity control', 'ppolag_fcsrl_capacity', 'e43/capacity_comparison.json'),
    ('Severity label', 'ppolag_fcsrl_severity', 'e45/comparison.json'),
    ('Cold-day sampling', 'ppolag_fcsrl_cold', 'e46/comparison.json'),
    ('Daily auxiliary target', 'ppolag_fcsrl_finite_day', 'e48/comparison.json'),
    ('Previous observation', 'ppolag_fcsrl_history', 'e54/history_comparison.json'),
    ('Repeated-current control', 'ppolag_fcsrl_repeated', 'e54/repeated_comparison.json'),
    ('One-step correction', 'ppolag_fcsrl_one_step_correction', 'e55/comparison.json'),
    ('Four-step rate correction', 'ppolag_fcsrl_rate_h4', 'e56/h4_comparison.json'),
    ('One-step regression control', 'ppolag_fcsrl_rate_h1', 'e56/h1_comparison.json'),
]
KEYS = ('cost', 'nh4_viol_pct', 'do_viol_pct')


def load_reports():
    complete, pending = [], []
    for name, method, relative in TRIALS:
        path = ROOT / 'experiments' / relative
        if not path.exists():
            raise FileNotFoundError(f'Previously completed trial is missing: {path}')
        report = json.loads(path.read_text())
        assert report['candidate_method'] == method and report['baseline_method'] == 'ppolag_fcsrl'
        assert report['policy_seeds'] == list(range(5)) and not report['test_evaluated']
        assert report['evaluation_environment'] == report['baseline_training_environment'] == report['candidate_training_environment'] == 'new'
        rows = report['metrics']['candidate_matched']
        air = rows['air_m3']
        assert np.all(np.abs(np.array(air['candidate_values']) - air['baseline_values']) <= .01 * np.array(air['baseline_values']))
        if complete:
            np.testing.assert_array_equal(air['baseline_values'], complete[0][1]['metrics']['candidate_matched']['air_m3']['baseline_values'])
        for key in KEYS:
            row = rows[key]
            difference = np.array(row['candidate_values']) - row['baseline_values']
            np.testing.assert_allclose(difference, row['paired_difference'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(difference.mean(), row['paired_mean_difference'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.std(row['candidate_values'], ddof=1), row['candidate_sd'], atol=1e-12, rtol=0)
            if complete:
                np.testing.assert_array_equal(row['baseline_values'], complete[0][1]['metrics']['candidate_matched'][key]['baseline_values'])
        if complete:
            assert report['freeze_canonical_sha256'] == complete[0][1]['freeze_canonical_sha256']
        expected = {'all_five_seeds_air_matched': True,
                    'cost_difference_interval_below_zero': rows['cost']['paired_seed_bootstrap95_ci'][1] < 0,
                    'nh4_difference_interval_below_zero': rows['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1] < 0,
                    'do_difference_interval_not_above_zero': rows['do_viol_pct']['paired_seed_bootstrap95_ci'][1] <= 0}
        assert report['checks'] == expected and report['meets_prespecified_simulation_criterion'] == all(expected.values())
        complete.append((name, report, relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return complete, pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    args = parser.parse_args()
    complete, pending = load_reports()
    assert complete
    fig, axes = plt.subplots(1, 3, figsize=(13, max(5.7, len(complete) * .57)), sharey=True)
    labels = ('Summed cost difference', 'NH4 difference (percentage points)', 'DO difference (percentage points)')
    for ax, key, label in zip(axes, KEYS, labels):
        ax.axvline(0, color='.4', linestyle=':', linewidth=1.1)
        for index, (_, report, _, _) in enumerate(complete):
            row = report['metrics']['candidate_matched'][key]
            low, high = row['paired_seed_bootstrap95_ci']
            ax.scatter(row['paired_difference'], index + np.linspace(-.10, .10, 5), color='.65', s=20)
            ax.plot([low, high], [index, index], color='#176f9b', linewidth=2.3)
            ax.scatter(row['paired_mean_difference'], index, color='#073d59', marker='D', s=35, zorder=4)
        ax.set_xlabel(label)
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(complete)), [name for name, _, _, _ in complete])
    axes[0].invert_yaxis()
    fig.suptitle('Completed agent trials at matched air: candidate minus FCSRL', fontsize=14)
    fig.text(.5, .06, 'Grey dots: five policy seeds. Diamond and line: mean and paired 95% seed-bootstrap interval.', ha='center', fontsize=9)
    successes = sum(report['meets_prespecified_simulation_criterion'] for _, report, _, _ in complete)
    fig.text(.5, .025, f'{successes}/{len(complete)} complete criteria passed. Reused validation and frozen surrogate; no independent confirmation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    text = ['# Completed fixed-environment agent trials', '',
            'All completed registered reports are included, including capacity/repeated-current controls and failures. These are exploratory reused-validation comparisons, not a ranking or independent confirmation.', '',
            'All methods use policy seeds0..4 and401280 transitions per seed, the same frozen e40 environment, final checkpoints and actual air matched within1 percent for each seed. Baseline air/cost/NH4/DO values and freeze identities agree across every report.', '',
            '![All completed agent trials](figures/' + args.out.name + ')', '',
            '| Candidate | Cost mean | NH4 percent | DO percent | Cost difference95 interval | NH4 difference95 interval | DO difference95 interval | Complete criterion |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, report, relative, _ in complete:
        rows = report['metrics']['candidate_matched']
        means = [f"{rows[key]['candidate_mean']:.6f}" for key in KEYS]
        intervals = [f"[{rows[key]['paired_seed_bootstrap95_ci'][0]:.6f},{rows[key]['paired_seed_bootstrap95_ci'][1]:.6f}]" for key in KEYS]
        passed = 'Pass' if report['meets_prespecified_simulation_criterion'] else 'Fail'
        text.append('| [' + name + '](../experiments/' + relative + ') | ' + ' | '.join([*means, *intervals, passed]) + ' |')
    text.extend(['', 'Differences are candidate minus its paired FCSRL reference; negative quality differences favor the candidate. NH4/DO intervals use percentage points. Cost is the95-step sum of the two violation indicators, not electricity or money. The complete criterion requires cost and NH4 upper endpoints below zero, DO upper endpoint at most zero and all five air matches.', '',
                 'Intervals use10000 paired policy-seed resamples. Model members, noise draws, dates and supervised classifier seeds are not independent policy seeds. Shared references correlate the trials. An interval crossing zero proves neither improvement nor equivalence. Multiple reused-validation comparisons are not multiplicity-adjusted independent evidence.', '',
                 'Native metrics, all air-only queries, sample SDs and seed values remain in the linked full reports/artifact indexes. The plot does not replace their source/checkpoint/replay guards. The e47 PPO diagnostic is documented separately because it is not a new FCSRL extension.', '',
                 'Pending registered reports: ' + (', '.join(pending) if pending else 'none') + '. No partial seed or smoke outcome is inserted as a complete result.', '',
                 '## Report checksums', ''])
    for name, _, relative, digest in complete:
        text.append(f'- {name}, {relative}, SHA256 {digest}')
    text.extend(['', 'Regenerate with Lee/analysis/plot_all_agent_trials.py. All inputs are compact generated summaries; no source observations, stochastic banks or final test metrics are read.', ''])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'completed': len(complete), 'complete_criteria_passed': successes, 'pending': pending}), flush=True)


if __name__ == '__main__':
    main()
