"""Authenticate immutable e63 matched policies without new simulator queries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e63'))
from trust_policy import load_policy
from rate_io import load_base, load_rates
from policy_io import verify_sources
from quantile_env import FrozenDynamics, frozen_digest
from report import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'study', 'policies', 'baselines', 'rates', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    frozen = FrozenDynamics(args.freeze)
    assert frozen.manifest['test_evaluated'] is False
    status = json.loads((args.study / 'study_status.json').read_text())
    manifest = json.loads((args.study / 'study_manifest.json').read_text())
    assert status['phase'] == 'complete' and not status['running'] and not status['pending']
    verify_sources(manifest['source_sha256'], ROOT)
    records, pairs = [], []
    for seed in range(5):
        run = args.policies / f'seed_{seed}'
        comparison = args.study / f'comparisons/seed_{seed}'
        pair = json.loads((comparison / 'summary.json').read_text())
        audit = json.loads((comparison / 'action_audit.json').read_text())
        assert pair['policy_seed'] == audit['policy_seed'] == seed
        assert pair['candidate_method'] == 'ppolag_fcsrl_risk_only_h4' and pair['penalty'] == 0
        assert pair['training_steps'] == 401280 and not pair['smoke_only'] and not pair['pipeline_smoke_only']
        assert not pair['test_evaluated'] and pair['checkpoint_replay_verified']
        assert pair['freeze_canonical_sha256'] == frozen_digest(args.freeze)
        verify_sources(pair['comparison_sources_sha256'], ROOT / 'Lee/experiments/e63')
        assert audit['passed'] and set(audit['replays']) == {'native', 'matched'}
        assert audit['comparison_summary_sha256'] == sha(comparison / 'summary.json')
        for replay in audit['replays'].values():
            assert replay['action_rows'] == 152000
            assert replay['independent_sklearn_actions_exact'] and replay['trajectory_replay_exact']
            assert replay['executed_air_accounting_exact']
        base, config, _, hashes = load_base(run, args.freeze, args.baselines / f'seed_{seed}')
        folder = args.rates / f'models/h4/seed_{seed}'
        models, fit = load_rates(folder, config, 4, False)
        assert fit['data_sha256'] == sha(args.data)
        assert hashes == fit['base_checkpoint_hashes'] == pair['candidate_checkpoint_hashes']
        assert hashes == pair['baseline_checkpoint_hashes']
        assert fit['model_sha256'] == pair['correction_model_sha256'] == sha(folder / 'risk_models.pkl')
        policy_path = comparison / 'controller_config.json'
        assert sha(policy_path) == pair['controller_config_sha256']
        policy = load_policy(base, models, policy_path)
        assert policy.config['penalty'] == 0
        native = json.loads((comparison / 'candidate_native.json').read_text())
        matched = json.loads((comparison / 'candidate_matched.json').read_text())
        assert native['summary'] == pair['candidate_native'] and matched['summary'] == pair['candidate_matched']
        assert pair['matching']['matched_within_one_percent']
        records.append({'policy_seed': seed, 'base_checkpoint_hashes': hashes,
            'rate_model_sha256': fit['model_sha256'], 'controller_config_sha256': sha(policy_path),
            'comparison_summary_sha256': sha(comparison / 'summary.json'),
            'action_audit_sha256': sha(comparison / 'action_audit.json'),
            'native_file_sha256': sha(comparison / 'candidate_native.json'),
            'matched_file_sha256': sha(comparison / 'candidate_matched.json'),
            'fixed_e63_offset_after_wrapper': pair['matching']['offset'],
            'primary_reference_air_m3': pair['candidate_matched']['air_m3'],
            'primary_reference_metrics': pair['candidate_matched'],
            'e63_native_metrics': pair['candidate_native'], 'original_fcsrl_metrics': pair['baseline_native']})
        pairs.append(pair)
        print('BASELINE_AUTHENTICATED', seed, flush=True)
    generated = summarize(pairs)
    assert generated == json.loads((args.study / 'final_reports/comparison.json').read_text())
    assert generated == json.loads((ROOT / 'Lee/experiments/e63/comparison.json').read_text())
    assert generated['meets_prespecified_simulation_criterion']
    means = {key: float(np.mean([row['primary_reference_metrics'][key] for row in records]))
             for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct')}
    np.testing.assert_allclose([means[k] for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct')],
        [52018.53777884504, 21.060125, 21.84473684210526, .3238157894736842], rtol=1e-12)
    result = {'passed': True, 'baseline': 'e63 matched policy including fixed post-wrapper offset',
              'policy_seeds': list(range(5)), 'records': records, 'matched_reference_means': means,
              'freeze_canonical_sha256': frozen_digest(args.freeze), 'data_sha256': sha(args.data),
              'test_evaluated': False, 'new_model_fitting': False, 'new_environment_queries': 0,
              'source_identity_rule': 'Existing raw-or-canonical-LF digest guard for source only; exact model/data/config bytes.',
              'criterion': {'all_five_air_matches': True, 'cost_ci_upper_below_zero_vs_e63': True,
                            'nh4_ci_upper_below_zero_vs_e63': True, 'do_ci_upper_at_most_zero_vs_e63': True,
                            'bootstrap_resamples': 10000, 'bootstrap_rng': 2026},
              'source_sha256': sha(Path(__file__)),
              'scope': 'Baseline authentication only, not a candidate result or new independent confirmation.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('BASELINE_AUTHENTICATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
