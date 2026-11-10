"""재학습 없이 DO 모델 교체의 NH4 재생 영향과 실측 DO 입력 진단을 분리한다."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from mapping import STATE
from common import save, sha
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import summarize


def replay(days, split, dynamics, observed_do=False):
    env = BatchEnv(days, dynamics)
    env.reset(all_episodes(split, dynamics.n_members))
    actual = np.stack([days[d]['state'][1:] for d, _ in env.episodes])
    preds = []
    for _ in range(env.T):
        # Diagnostic oracle only, never supplied to a trained policy or deployment claim.
        if observed_do:
            env.state[:, 2:] = env.x['state'][:, env.t, 2:]
        env.step(env.historical_action())
        preds.append(env.state.copy())
    pred = np.stack(preds, 1)
    return summarize(actual, pred), pred


def main(a):
    splits, days = load_days(a.prepared / 'data.pkl')
    old = Dynamics(a.prepared / 'surrogate.pkl')
    new = Dynamics(a.root / 'corrected_surrogate.pkl')
    comparison = json.loads((a.root / 'comparison.json').read_text())
    results = {k: v['rollout_member_average'] for k, v in comparison['results'].items()}
    for j in (2, 3, 4):
        dyn = copy.copy(old)
        dyn.models = [[new.models[m][i] if i == j else model for i, model in enumerate(row)] for m, row in enumerate(old.models)]
        results['only_' + STATE[j]], _ = replay(days, splits['val'], dyn)
        print('PROBE', STATE[j], results['only_' + STATE[j]]['mae']['nh4'], flush=True)
    results['oracle_DO_legacy'], p = replay(days, splits['val'], old, observed_do=True)
    results['oracle_DO_corrected'], q = replay(days, splits['val'], new, observed_do=True)
    assert np.array_equal(p[:, :, :2], q[:, :, :2])
    save(a.root / 'feedback_probe.json', {'results': results,
         'oracle_nh4_no3_exactly_equal': True, 'policy_training': False, 'test_evaluated': False,
         'script_sha256': sha(Path(__file__)),
         'model_sha256': {'legacy': sha(a.prepared / 'surrogate.pkl'), 'corrected': sha(a.root / 'corrected_surrogate.pkl')},
         'interpretation': 'Model-path diagnostic with recorded actions. Observed future DO is an evaluation oracle, not an available policy input. Single swaps do not add up when interactions exist.'})
    print('DONE', results['oracle_DO_corrected']['mae'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args())
