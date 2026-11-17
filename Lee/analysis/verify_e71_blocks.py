"""Recompute episode aggregates, air-only decisions and separate seed-block criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
from air_match import match_air


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-fresh', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    results = workspace / 'results'
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    days = pd.read_pickle(data)['days']
    validation = {str(day) for day in days['val']}
    assert len(validation) == 40 and len(days['test']) == 41
    expected_keys = {(day, member, draw) for day in validation for member in range(5) for draw in range(8)}
    inputs, blocks = {}, []
    indices = np.random.default_rng(2026).integers(0, 5, (10000, 5))

    def checked_read(path):
        inputs[str(path.relative_to(workspace))] = sha(path)
        return read(path)

    def episodes(path):
        value = checked_read(path)
        rows, summary = value['rows'], value['summary']
        assert len(rows) == len(expected_keys) == 1600
        assert {(row['day'], row['member'], row['noise_draw']) for row in rows} == expected_keys
        for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct'):
            values = np.array([row[key] for row in rows])
            assert np.isfinite(values).all()
            np.testing.assert_allclose(values.mean(), summary[key], rtol=0, atol=1e-10)
        assert all(0 <= row['air_m3'] <= 213750 and 0 <= row['cost'] <= 190 for row in rows)
        for row in rows:
            np.testing.assert_allclose(row['cost'], .95 * (row['nh4_viol_pct'] + row['do_viol_pct']), atol=1e-10, rtol=0)
        np.testing.assert_allclose(summary['reward'], -summary['air_m3'] / 2250, atol=1e-10, rtol=0)
        np.testing.assert_allclose(summary['days_within_limit_pct'], np.mean([row['cost'] <= 12 for row in rows]) * 100, atol=1e-10, rtol=0)
        assert summary['n_episodes'] == 1600
        return summary

    for label, seeds, name, cert_path in (
        ('original', list(range(5)), 'e71_study', ROOT / 'Lee/experiments/e64/baseline_authentication.json'),
        ('fresh', list(range(5, 10)), 'e71_fresh_study', results / 'e68_fresh_references/baseline_authentication.json')):
        study = results / name
        if not (study / 'comparison.json').exists():
            blocks.append({'block': label, 'complete': False, 'criterion_passed': None})
            assert label == 'fresh' and not args.require_fresh, 'Required block is incomplete'
            continue
        status = checked_read(study / 'study_status.json')
        assert status['phase'] == 'complete' and not status['running'] and not status['pending']
        cert, report = checked_read(cert_path), checked_read(study / 'comparison.json')
        assert cert['policy_seeds'] == report['policy_seeds'] == seeds
        assert cert['authentication_passed' if label == 'fresh' else 'passed']
        assert not report['test_evaluated']
        all_matches, pairs = [], []
        for seed in seeds:
            folder = study / f'comparisons/seed_{seed}'
            pair, audit = checked_read(folder / 'summary.json'), checked_read(folder / 'action_audit.json')
            reference = next(row for row in cert['records'] if row['policy_seed'] == seed)
            assert pair['policy_seed'] == audit['policy_seed'] == seed and audit['passed']
            assert not pair['test_evaluated'] and not audit['test_evaluated']
            assert pair['training_steps'] == 401280 and pair['extra_environment_training_steps'] == 0
            assert pair['baseline_certificate_sha256'] == sha(cert_path)
            assert audit['comparison_summary_sha256'] == sha(folder / 'summary.json')
            assert pair['baseline_native'] == reference['primary_reference_metrics']
            assert pair['base_checkpoint_hashes'] == reference['base_checkpoint_hashes']
            assert pair['correction_model_sha256'] == reference['rate_model_sha256']
            assert pair['baseline_fixed_offset'] == reference['fixed_e63_offset_after_wrapper']
            assert episodes(folder / 'baseline_native.json') == pair['baseline_native']
            for mode in ('native', 'matched'):
                assert episodes(folder / f'candidate_{mode}.json') == pair[f'candidate_{mode}']
                replay = audit['replays'][mode]
                assert replay['action_rows'] == 152000 and replay['trajectory_replay_exact']
                assert replay['independent_sklearn_actions_exact'] and replay['executed_air_accounting_exact']
            calls = []
            def air_only(offset):
                calls.append(offset)
                volume = episodes(folder / f'query_{len(calls):02d}.json')['air_m3']
                return volume
            matching = match_air(air_only, pair['baseline_native']['air_m3'], pair['candidate_native']['air_m3'])
            assert matching == pair['matching'] and len(calls) <= 25
            assert matching['achieved_air_m3'] == pair['candidate_matched']['air_m3']
            all_matches.append(matching['matched_within_one_percent'])
            pairs.append(pair)
        for mode in ('candidate_native', 'candidate_matched'):
            for key, row in report['metrics'][mode].items():
                baseline = np.array([pair['baseline_native'][key] for pair in pairs])
                candidate = np.array([pair[mode][key] for pair in pairs])
                delta = candidate - baseline
                expected = {'baseline_mean': baseline.mean(), 'candidate_mean': candidate.mean(),
                    'baseline_sd': baseline.std(ddof=1), 'candidate_sd': candidate.std(ddof=1),
                    'baseline_values': baseline, 'candidate_values': candidate, 'paired_difference': delta,
                    'paired_mean_difference': delta.mean(),
                    'paired_seed_bootstrap95_ci': np.quantile(delta[indices].mean(1), [.025, .975])}
                for field, value in expected.items():
                    np.testing.assert_allclose(value, row[field], atol=1e-10, rtol=0)
        matched = report['metrics']['candidate_matched']
        checks = {'all_five_seeds_air_matched': all(all_matches),
            'cost_difference_interval_below_zero': matched['cost']['paired_seed_bootstrap95_ci'][1] < 0,
            'nh4_difference_interval_below_zero': matched['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1] < 0,
            'do_difference_interval_not_above_zero': matched['do_viol_pct']['paired_seed_bootstrap95_ci'][1] <= 0}
        assert checks == report['checks']
        assert all(checks.values()) == report['meets_prespecified_simulation_criterion']
        blocks.append({'block': label, 'policy_seeds': seeds, 'complete': True,
                       'criterion_passed': all(checks.values()), 'checks': checks})
    result = {'blocks': blocks, 'both_blocks_verified_and_passed': all(row['complete'] and row['criterion_passed'] for row in blocks),
        'data_sha256': sha(data), 'input_sha256': inputs, 'verifier_sha256': sha(Path(__file__)),
        'new_environment_queries': 0, 'test_evaluated': False,
        'scope': 'Independent recorded episode/air-only matching/paired-seed arithmetic checks. No block pooling or goal completion inferred from an incomplete block.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(blocks), flush=True)


if __name__ == '__main__':
    main()
