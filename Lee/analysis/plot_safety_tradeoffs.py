"""Summarize audited environment tradeoffs without reading plant observations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / 'Lee/experiments'


def read(path):
    value = json.loads(path.read_text())
    assert value['test_evaluated'] is False
    return value


def interval(row):
    return row.get('five_day_block_95_ci', row.get('circular_five_day_bootstrap95_ci'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    args = parser.parse_args()
    definitions = [('e40', 'frozen', 'event_reference_audit.json', 'e40'),
                   ('e41', 'history', 'event_reference_audit.json', 'e41'),
                   ('e50', 'exposure', 'event_reference_audit.json', 'candidate'),
                   ('e51', 'matched PIT', 'event_reference_audit.json', 'candidate'),
                   ('e52', 'tail noise', 'audit.json', 'candidate')]
    data = []
    for experiment, label, event_file, event_key in definitions:
        folder = EXPERIMENTS / experiment
        summary = read(folder / 'summary.json')
        score = summary['validation'][summary['selected']] if 'selected' in summary else summary['validation']
        audit, event = read(folder / 'audit.json'), read(folder / event_file)
        events = event['metrics'][event_key]
        for constraint in ('nh4', 'do'):
            np.testing.assert_allclose(events[constraint]['brier'], score[constraint + '_brier'], rtol=1e-12)
            np.testing.assert_allclose(events[constraint]['prevalence'] * 100,
                                       score[constraint + '_violation']['actual'], rtol=1e-12)
            expected_onsets = 89 if constraint == 'nh4' else 25
            assert events[constraint]['onset_tolerance_steps']['4']['actual_onsets'] == expected_onsets
            np.testing.assert_allclose(audit['date_intervals'][constraint + '_bias_pp']['mean'],
                score[constraint + '_violation']['pred'] - score[constraint + '_violation']['actual'], atol=1e-10)
        difference = ({constraint: {'mean': 0., 'circular_five_day_bootstrap95_ci': [0., 0.]}
                       for constraint in ('nh4', 'do')} if experiment == 'e40'
                      else event['paired_date_brier_differences']['e40'])
        data.append({'id': experiment, 'label': label, 'summary': summary, 'score': score,
                     'audit': audit, 'events': events, 'difference': difference})
    reference = data[0]['audit']['reference_scores']['old']
    tolerance = {'nh4': min(5., reference['nh4_violation']['absolute_error_pp']),
                 'do': min(1., reference['do_violation']['absolute_error_pp'])}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    x = np.arange(len(data))
    labels = [row['id'] + '\n' + row['label'] for row in data]
    for column, constraint in enumerate(('nh4', 'do')):
        ax = axes[0, column]
        means = np.array([row['audit']['date_intervals'][constraint + '_bias_pp']['mean'] for row in data])
        limits = np.array([interval(row['audit']['date_intervals'][constraint + '_bias_pp']) for row in data])
        ax.axhspan(-tolerance[constraint], tolerance[constraint], color='#e9f1e8', label='Point frequency tolerance')
        ax.axhline(0, color='#555555', linewidth=1)
        ax.errorbar(x, means, yerr=np.stack([means - limits[:, 0], limits[:, 1] - means]),
                    fmt='o', capsize=4, color='#265b87', linewidth=1.5)
        ax.set_title(constraint.upper() + ' frequency bias')
        ax.set_ylabel('Predicted minus observed, percentage points')
        ax.set_xticks(x, labels)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper left', fontsize=8)
        ax = axes[1, column]
        means = np.array([row['difference'][constraint]['mean'] for row in data])
        limits = np.array([row['difference'][constraint]['circular_five_day_bootstrap95_ci'] for row in data])
        ax.axhline(0, color='#555555', linewidth=1)
        ax.errorbar(x, means, yerr=np.stack([means - limits[:, 0], limits[:, 1] - means]),
                    fmt='o', capsize=4, color='#265b87', linewidth=1.5)
        ax.set_title(constraint.upper() + ' probability error difference')
        ax.set_ylabel('Brier difference from e40, lower is better')
        ax.set_xticks(x, labels)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Environment candidates: frequency and probability tradeoffs', fontsize=14, y=.98)
    fig.text(.5, .035, '40 reused validation dates, circular five-day intervals. Shading applies to point estimates only.',
             ha='center', fontsize=9)
    fig.text(.5, .012, 'No candidate matches a new DO onset within one hour (25 observed). These are not policy-seed comparisons.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95), h_pad=2.)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    lines = ['# Audited environment tradeoffs', '',
        'This table compares environment forecasts only. All models use the same40 reused validation dates.',
        'No policy result or final test observation enters this summary. Smaller MAE or closer frequency alone',
        'does not establish a complete safety-prediction improvement.', '',
        '| Model | Normalized MAE | NH4 predicted percent | DO predicted percent | NH4 Brier | DO Brier | NH4 onsets within1h | DO onsets within1h | Original point gate |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in data:
        score, events = row['score'], row['events']
        name = row['id']
        lines.append(f"| [{name}](../experiments/{name}/summary.json) | {score['normalized_mae']:.6f} | "
            f"{score['nh4_violation']['pred']:.6f} | {score['do_violation']['pred']:.6f} | "
            f"{score['nh4_brier']:.6f} | {score['do_brier']:.6f} | "
            f"{events['nh4']['onset_tolerance_steps']['4']['matched']}/89 | "
            f"{events['do']['onset_tolerance_steps']['4']['matched']}/25 | "
            f"{'pass' if row['summary']['gate']['passed'] else 'fail'} |")
    lines.extend(['', 'Actual NH4/DO frequency is17.315789/1.157895 percent. The original joint point gate',
        'also checks errors, member agreement and common-noise action ordering. It is not a population',
        'calibration guarantee or a DO-onset criterion.', '',
        '![Audited safety tradeoffs](figures/' + args.out.name + ')', '',
        'Intervals describe date variation, with model/noise draws averaged within dates. Models and draws',
        'are not independent policy seeds. Repeated validation and multiple hypotheses remain exploratory.',
        'e40 stays frozen for agent comparisons; e51 is a partial frequency/timing change and e52 fails.',
        'No single candidate establishes accurate rare DO onset prediction. Full audits, source provenance',
        'and artifact locations remain in each experiment folder.', '',
        'Regenerate with plot_safety_tradeoffs.py using only committed compact reports; raw data are not needed.', ''])
    args.table.write_text('\n'.join(lines))
    print(json.dumps({'figure': str(args.out), 'table': str(args.table), 'test_evaluated': False, 'policy_results_included': False}), flush=True)


if __name__ == '__main__':
    main()
