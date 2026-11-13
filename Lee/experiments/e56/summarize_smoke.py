"""Require both regression smoke pipelines before the full fixed-horizon study."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    preflight = json.loads((here / 'model_check.json').read_text())
    assert preflight['passed']
    rows = []
    identities = []
    captures = []
    for horizon in (1, 4):
        fit = json.loads((args.root / f'h{horizon}/fit_summary.json').read_text())
        held = json.loads((args.root / f'h{horizon}/held_audit.json').read_text())
        pair = json.loads((args.root / f'comparison_h{horizon}/summary.json').read_text())
        actions = json.loads((args.root / f'comparison_h{horizon}/action_audit.json').read_text())
        assert fit['horizon'] == pair['correction_horizon'] == horizon
        assert fit['policy_seed'] == pair['policy_seed'] == 0
        assert fit['smoke_only'] and pair['smoke_only'] and held['smoke_only'] and actions['smoke_only']
        assert fit['fit_transitions'] == fit['held_transitions'] == 6080
        assert fit['base_training_steps'] == pair['training_steps'] == 401280
        assert fit['fit_updates'] == [1] and fit['held_updates'] == [2]
        assert fit['checks']['passed'] and held['passed'] and actions['passed']
        assert not pair['test_evaluated'] and not fit['test_evaluated']
        assert pair['checkpoint_replay_verified']
        assert pair['candidate_checkpoint_hashes'] == pair['baseline_checkpoint_hashes'] == fit['base_checkpoint_hashes']
        assert pair['candidate_method'] == f'ppolag_fcsrl_rate_h{horizon}'
        assert all(pair['checks'].values())
        assert pair['matching']['matched_within_one_percent']
        target, actual = pair['baseline_native']['air_m3'], pair['candidate_matched']['air_m3']
        assert abs(actual - target) <= target * .01
        assert actions['actions']['independent_sklearn_choices_exact']
        assert actions['actions']['native_replay_exact'] and actions['actions']['action_rows'] == 152000
        if horizon == 4:
            assert actions['actions']['flat_rate_null_full_day_exact']
        rows.append({'horizon': horizon, 'fit_transitions': 6080, 'held_transitions': 6080,
                     'base_policy_training_steps': 401280, 'inference_checks': fit['checks'],
                     'held_metrics_reconstructed': held['held']['held_metrics_reconstructed'],
                     'native_replay_checks': pair['checks'], 'matching': pair['matching'],
                     'action_audit': actions['actions']})
        identities.append(pair['baseline_checkpoint_hashes'])
        captures.append(fit['capture_file_sha256'])
    assert identities[0] == identities[1] and captures[0] == captures[1]
    result = {'passed': True, 'smoke_only': True, 'test_evaluated': False,
              'policy_seed': 0, 'horizons': rows, 'same_base_and_capture_subsets_verified': True,
              'full_horizon_policy_seed_results_completed': [],
              'runtime_sources': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                  for p in here.glob('*.py')},
              'scope': 'Both small regressor fits on update1 with update2 held, attached to an existing full-budget base policy. Not five complete regressor/control trials. Batch orchestration is compiled/source pinned; underlying operations are exercised.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('E56_SMOKE_PIPELINE_PASS', flush=True)


if __name__ == '__main__':
    main()
