"""Independent public-feature and action-choice audit, without fitting or tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from policy_io import load_captured, load_correction, corrected_policy
from quantile_env import FrozenDynamics, evaluate
from cost_model import features


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'baseline', 'candidate', 'correction', 'comparison', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    base, config, _, _ = load_captured(args.candidate, args.freeze, args.baseline, args.allow_smoke)
    models, _ = load_correction(args.correction, config, args.allow_smoke)
    policy = corrected_policy(base, models)
    # Direct independent construction of the fixed candidate scores.
    counts = np.zeros(5, dtype=int)
    totals = {'actions': 0, 'changed': 0, 'candidates_clipped': 0,
              'predicted_risk_difference_sum': 0., 'squared_adjustment_sum': 0.}
    def audited_action(obs):
        original = np.asarray(base.act_det(obs)).reshape(-1)
        raw = original[:, None] + np.array([0, -.125, .125, -.25, .25])
        actions = np.clip(raw, -1, 1)
        inputs = np.column_stack((np.repeat(obs, 5, axis=0), actions.reshape(-1)))
        inputs[:, 1] = 0
        np.testing.assert_array_equal(inputs, features(np.repeat(obs, 5, axis=0), actions.reshape(-1)))
        probability = sum(model.predict_proba(inputs)[:, 1] for model in models).reshape(-1, 5)
        score = probability + np.square(actions - original[:, None])
        chosen = np.argmin(score, axis=1)
        expected = actions[np.arange(len(obs)), chosen]
        actual = policy.act_det(obs)
        np.testing.assert_array_equal(actual, expected)
        counts[:] += np.bincount(chosen, minlength=5)
        totals['actions'] += len(obs)
        totals['changed'] += int(np.count_nonzero(actual != original))
        totals['candidates_clipped'] += int(np.count_nonzero(actions != raw))
        totals['predicted_risk_difference_sum'] += float((probability[np.arange(len(obs)), chosen] - probability[:, 0]).sum())
        totals['squared_adjustment_sum'] += float(np.square(actual - original).sum())
        return actual
    actual, rows = evaluate(args.data, FrozenDynamics(args.freeze), act_batch=audited_action)
    stored = json.loads((args.comparison / 'candidate_native.json').read_text())
    assert stored['summary'] == actual and stored['rows'] == rows
    n = totals['actions']
    result = {'passed': True, 'policy_seed': config['seed'], 'test_evaluated': False,
              'smoke_only': args.allow_smoke, 'native_replay_exact': True,
              'independent_sklearn_choice_exact': True, 'action_count': n,
              'changed_action_fraction': totals['changed'] / n,
              'candidate_choice_counts': counts.tolist(),
              'candidate_clipped_fraction': totals['candidates_clipped'] / (5 * n),
              'mean_predicted_risk_change': totals['predicted_risk_difference_sum'] / n,
              'mean_squared_adjustment': totals['squared_adjustment_sum'] / n,
              'scope': 'Native final-controller public-feature audit, no fitting or parameter selection. Predicted risk improvement is not measured water-quality improvement.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('CORRECTION_ACTION_AUDIT_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
