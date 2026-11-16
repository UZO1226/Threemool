"""Independent sklearn replay for every native and matched fresh comparator."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_reference import ROOT, sha, load_base, load_rates, load_policy, verify_sources
from quantile_env import FrozenDynamics, evaluate, frozen_digest
from rate_model import ConstantRate
from trust_policy import TrustPolicy


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'baselines', 'rates', 'reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--null-check', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    summary = json.loads((args.reference / 'summary.json').read_text())
    assert summary['method'] == 'e63_matched_risk_only_h4'
    assert not summary['test_evaluated'] and not summary['candidate_evaluation']
    seed = summary['policy_seed']
    assert seed in range(5, 10) and summary['base_training_steps'] == 401280
    verify_sources(summary['source_sha256'], Path(__file__).parent)
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    assert frozen_digest(freeze) == summary['freeze_canonical_sha256']
    data = args.workspace.resolve() / 'kaggle_data/darrow_15min.pkl'
    assert sha(data) == summary['data_sha256']
    base, config, _, hashes = load_base(args.baselines / f'captured/new_ppolag_fcsrl/seed_{seed}',
        freeze, args.baselines / f'original/new_ppolag_fcsrl/seed_{seed}')
    models, fit = load_rates(args.rates / f'models/h4/seed_{seed}', config, 4, False)
    assert hashes == summary['base_checkpoint_hashes']
    assert fit['model_sha256'] == summary['rate_model_sha256']
    assert sha(args.reference / 'controller_config.json') == summary['controller_config_sha256']
    policy = load_policy(base, models, args.reference / 'controller_config.json')
    dynamics = FrozenDynamics(freeze)
    replays = {}
    for phase, offset in [('native', 0.), ('matched', summary['matching']['offset'])]:
        totals = {'rows': 0, 'volume': 0.}
        def act(obs):
            preserved = obs.copy()
            original = np.asarray(base.act_det(obs)).reshape(-1)
            candidates = np.clip(original[:, None] + [0, -.125, .125, -.25, .25], -1, 1)
            x = np.column_stack((np.repeat(obs, 5, axis=0), candidates.reshape(-1)))
            rate = np.clip(np.column_stack([model.predict(x) for model in models]), 0, 1)
            selected = rate.sum(-1).reshape(-1, 5).argmin(-1)
            expected = candidates[np.arange(len(obs)), selected]
            actual = policy.act_det(obs)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(obs, preserved)
            assert np.max(np.abs(actual - original)) <= .25 + 1e-12
            totals['rows'] += len(obs)
            executed = np.clip(actual + offset, -1, 1)
            totals['volume'] += float(np.sum((executed + 1) / 2 * 9000 * .25))
            return actual
        replay, rows = evaluate(data, dynamics, act_batch=act, air_offset=offset)
        saved = json.loads((args.reference / f'e63_{phase}.json').read_text())
        assert saved == {'summary': replay, 'rows': rows}
        assert totals['rows'] == 152000
        np.testing.assert_allclose(totals['volume'] / 1600, replay['air_m3'], rtol=1e-12, atol=1e-8)
        replays[phase] = {'independent_sklearn_actions_exact': True, 'trajectory_replay_exact': True,
            'executed_air_accounting_exact': True, 'action_rows': totals['rows'], 'offset': offset}
        print('FRESH_REFERENCE_AUDIT', seed, phase, flush=True)
    if args.null_check:
        null = TrustPolicy(base, [ConstantRate(.3), ConstantRate(.8)], 0)
        replay, rows = evaluate(data, dynamics, act_batch=null.act_det)
        assert json.loads((args.reference / 'original_fcsrl_native.json').read_text()) == {'summary': replay, 'rows': rows}
    result = {'passed': True, 'policy_seed': seed, 'replays': replays,
        'flat_risk_null_full_native_exact': args.null_check, 'test_evaluated': False,
        'summary_sha256': sha(args.reference / 'summary.json'), 'audit_source_sha256': sha(Path(__file__)),
        'scope': 'Comparator authentication only, not candidate replication or physical safety.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('FRESH_REFERENCE_AUDIT_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    main()
