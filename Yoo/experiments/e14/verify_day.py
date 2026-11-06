"""하루 gradient, 조건별 제약 및 누락 위반 proxy 제거를 확인한다."""
import copy
import json
import torch
from day_trial import CFG, group_terms, objectives, dual_step, penalty, tradeoff_gate


class Accumulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.001))
        self.delta_sd = torch.ones(2)

    def forward(self, state, exog, extra):
        return state + self.weight


def main():
    torch.set_num_threads(1)
    h = 275
    model = Accumulator()
    state = torch.zeros(2, h+1, 2)
    exog, extra = torch.zeros(2, h, 6), torch.zeros(2, h, 8)
    bounds = torch.tensor([10., 20.])
    day, anchor, _ = objectives(model, state, exog, extra, torch.ones(2), bounds, h)
    gd = torch.autograd.grad(day, model.weight, retain_graph=True)[0]
    ga = torch.autograd.grad(anchor, model.weight)[0]
    torch.testing.assert_close(gd, torch.tensor(.002*(h+1)*(2*h+1)/6), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ga, torch.tensor(.002))
    short, _, _ = objectives(model, state, exog, extra, torch.ones(2), bounds, 36)
    gs = torch.autograd.grad(short, model.weight)[0]
    expected = .002*sum((i % 36 + 1)**2 for i in range(h))/h
    torch.testing.assert_close(gs, torch.tensor(expected), rtol=1e-5, atol=1e-6)
    pred = torch.zeros(1, 1000, 2)
    pred[:, :40] = torch.tensor([5., 3.])
    exog = torch.zeros(1, 1000, 6)
    exog[..., 1] = 1
    exog[:, :80, 1] = 0
    exog[..., 2] = 15
    proxy, fp, count = group_terms(pred, torch.zeros_like(pred), exog, bounds, torch.ones(2))
    m, avg, init = dual_step(torch.zeros(7, 2), torch.zeros(7, 2), torch.zeros(7, 2, dtype=torch.bool), fp, count)
    assert (m[0] == 0).all() and (m[1] > 0).all()
    assert not init[4].any() and not init[6].any()
    assert penalty(proxy, m, count, 'day_group') > penalty(proxy, m, count, 'day_global')
    probe = torch.tensor([[[3., 1.], [5., 3.]]], requires_grad=True)
    target = torch.tensor([[[5., 3.], [3., 1.]]])
    args = (target, exog[:, :2], bounds, torch.ones(2))
    p, _, c = group_terms(probe, *args)
    zero = penalty(p, torch.zeros_like(p), c, 'day_group')
    g = torch.autograd.grad(zero, probe, retain_graph=True)[0]
    assert (g == 0).all()
    # 표본 최소 수와 별개로 FP proxy 자체는 실제 위반을 밀어 올리지 않는다.
    g = torch.autograd.grad(p.sum(), probe)[0]
    assert (g[0, 0] == 0).all() and (g[0, 1] > 0).all()
    all_positive = group_terms(probe, torch.full_like(probe, 10), exog[:, :2], bounds, torch.ones(2))
    assert all(torch.isfinite(v).all() for v in all_positive)
    rows = [dict(mode=mode, seed=seed, split='inner_holdout', reset_steps=275,
                 nh4=dict(step_recall=.5, fpr=.15 if mode != 'day_group' else .1))
            for mode in ('start_multi', 'day_mse', 'day_group') for seed in CFG['seeds']]
    assert tradeoff_gate(rows, 'day_group')['passed']
    missed = copy.deepcopy(rows)
    for r in missed:
        if r['mode'] == 'day_group': r['nh4']['step_recall'] = .49
    assert not tradeoff_gate(missed, 'day_group')['passed']
    json.dumps(tradeoff_gate(missed, 'day_group'), allow_nan=False)
    print('PASS 275-step gradients, short reset, hidden subgroup FPR, missing support, no FN proxy, zero penalty at zero lambda, recall gate')


if __name__ == '__main__':
    main()
