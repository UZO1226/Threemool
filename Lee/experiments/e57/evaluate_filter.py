"""Use the original evaluator while observing actual command variation separately."""
import numpy as np

from filter_io import FrozenDynamics
from quantile_env import evaluate


def score(data, dynamics, policy, offset=0.):
    assert isinstance(dynamics, FrozenDynamics)
    tracker = {'last': None, 'volume': None, 'absolute_change': 0., 'maximum_change': 0.,
               'adjacent_samples': 0, 'initial_jump': None, 'calls': 0,
               'previous_outside': 0, 'action_rows': 0}
    def observed(obs):
        action = np.asarray(policy.act_det(obs)).reshape(-1)
        # Match the original evaluator's dtype-preserving addition, then the
        # environment's float conversion, clipping and physical-air arithmetic.
        executed = np.clip(np.asarray(action + offset, dtype=float), -1, 1)
        air = (executed + 1) / 2 * 9000
        if tracker['last'] is None:
            tracker['volume'] = np.zeros(len(obs))
            tracker['initial_jump'] = float(np.mean(np.abs(air - obs[:, 10].astype(float) * 9000)))
        else:
            change = np.abs(air - tracker['last'])
            tracker['absolute_change'] += float(change.sum())
            tracker['maximum_change'] = max(tracker['maximum_change'], float(change.max()))
            tracker['adjacent_samples'] += len(obs)
        tracker['volume'] += air * .25
        tracker['last'] = air.copy()
        tracker['calls'] += 1
        tracker['previous_outside'] += int(np.count_nonzero((obs[:, 10] < 0) | (obs[:, 10] > 1)))
        tracker['action_rows'] += len(obs)
        return action
    summary, rows = evaluate(data, dynamics, act_batch=observed, air_offset=offset)
    assert tracker['calls'] == 95 and tracker['adjacent_samples'] == 94 * 1600
    np.testing.assert_allclose(tracker['volume'].mean(), summary['air_m3'], atol=1e-8, rtol=1e-12)
    diagnostics = {'mean_absolute_adjacent_air_change_m3_h': tracker['absolute_change'] / tracker['adjacent_samples'],
                   'maximum_absolute_adjacent_air_change_m3_h': tracker['maximum_change'],
                   'adjacent_change_samples': tracker['adjacent_samples'],
                   'initial_jump_from_public_previous_air_mean_m3_h': tracker['initial_jump'],
                   'public_previous_air_outside_command_range_fraction': tracker['previous_outside'] / tracker['action_rows'],
                   'executed_volume_reconstructed_tolerance_m3': 1e-8,
                   'scope': 'Command differences between executed doses, excluding the first jump. Initial jump uses the float32 public sensor. These are command-rate differences, not electrical energy or a new adoption criterion.'}
    return summary, rows, diagnostics
