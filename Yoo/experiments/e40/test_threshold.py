"""동점, 범위 밖 점수와 목표 경계를 완전탐색으로 검산한다."""
import numpy as np
from run import choose, apply


def exhaustive(score, truth, mode):
    candidates = [(0, 0, None)]
    for threshold in sorted(set(score), reverse=True):
        alerts = score >= threshold
        candidates.append((int((truth & alerts).sum()), int((~truth & alerts).sum()), threshold))
    if mode == 'recall80':
        candidates = [x for x in candidates if 5 * x[0] >= 4 * truth.sum()]
        return min(candidates, key=lambda x: (x[1], -float('inf') if x[2] is None else -x[2]))
    candidates = [x for x in candidates if 10 * x[1] <= (~truth).sum()]
    return min(candidates, key=lambda x: (-x[0], x[1], -float('inf') if x[2] is None else -x[2]))


def main():
    rng = np.random.default_rng(40)
    for i in range(500):
        scores = rng.integers(-20, 30, size=30).astype(float) if i % 2 else rng.normal(4, 10, 30)
        truth = rng.random(30) > .7
        truth[:2] = (True, False)
        for mode in ('recall80', 'fpr10'):
            op = choose(scores, truth, mode)
            expected = exhaustive(scores, truth, mode)
            assert (op['tp'], op['fp'], op['threshold']) == expected
    # All scores tied: a probability sentinel at 1 would incorrectly alert here.
    op = choose(np.full(20, 12.), np.arange(20) < 2, 'fpr10')
    assert op['rule'] == 'never_alert'
    assert not apply(np.array([-100., 12., 1e8]), op).any()
    # An exact 10% boundary is feasible and an indivisible larger group is not.
    truth = np.array([True] + [False] * 10)
    op = choose(np.array([9., 9.] + [0.] * 9), truth, 'fpr10')
    assert op['tp'] == 1 and op['fp'] == 1 and op['threshold'] == 9.
    op = choose(np.array([9., 9., 9.] + [0.] * 8), truth, 'fpr10')
    assert op['rule'] == 'never_alert'
    print('PASS: 1000 exhaustive comparisons and no-alert/tie/boundary cases')


if __name__ == '__main__':
    main()
