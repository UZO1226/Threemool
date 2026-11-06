"""제약의 방향과 분모, 연속 예측 gradient, 진행 조건을 확인한다."""
import copy
import json
import torch
from experiment import CFG, conditional_terms, dual_step, objectives, tradeoff_gate


class Accumulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.01))
        self.delta_sd = torch.ones(2)

    def forward(self, state, exog, extra):
        return state + self.weight


def main():
    torch.set_num_threads(1)
    pred = torch.tensor([[3., 1.], [5., 3.]], requires_grad=True)
    target = torch.tensor([[5., 3.], [3., 1.]])
    fn, fp, hard, valid = conditional_terms(pred, target, torch.ones(2))
    torch.testing.assert_close(hard, torch.ones(2))
    gfn = torch.autograd.grad(fn.sum(), pred, retain_graph=True)[0]
    gfp = torch.autograd.grad(fp.sum(), pred)[0]
    assert (gfn[0] < 0).all() and (gfn[1] == 0).all()
    assert (gfp[1] > 0).all() and (gfp[0] == 0).all()
    duplicate = conditional_terms(torch.cat((pred.detach(), pred.detach()[1:])),
                                  torch.cat((target, target[1:])), torch.ones(2))
    torch.testing.assert_close(duplicate[0], fn)
    torch.testing.assert_close(duplicate[1], fp)
    for target_value in (0., 10.):
        terms = conditional_terms(pred, torch.full_like(pred, target_value), torch.ones(2))
        assert all(torch.isfinite(v).all() for v in terms)
    multiplier, average = dual_step(torch.ones(2), None, torch.tensor([.2, .01]), valid)
    assert multiplier[0] > 1 and multiplier[1] < 1 and not multiplier.requires_grad
    m, _ = dual_step(torch.zeros(2), None, torch.zeros(2), valid)
    assert (m == 0).all()
    m, _ = dual_step(torch.ones(2), None, torch.ones(2), torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(m, torch.ones(2))
    h = CFG['horizon']
    model = Accumulator()
    multi, anchor, _ = objectives(model, torch.zeros(2, h+1, 2), torch.zeros(2, h, 6),
                                   torch.zeros(2, h, 8), torch.ones(2))
    gm = torch.autograd.grad(multi, model.weight, retain_graph=True)[0]
    ga = torch.autograd.grad(anchor, model.weight)[0]
    torch.testing.assert_close(gm, torch.tensor(.02 * (h+1) * (2*h+1) / 6))
    torch.testing.assert_close(ga, torch.tensor(.02))
    rows = [dict(mode=mode, seed=seed, split='inner_holdout', reset_steps=275,
                 nh4=dict(step_recall=.5, fpr=.15 if mode != 'candidate' else .1))
            for mode in ('start_multi', 'multi', 'candidate') for seed in CFG['seeds']]
    assert tradeoff_gate(rows, 'candidate')['passed']
    missed = copy.deepcopy(rows)
    for row in missed:
        if row['mode'] == 'candidate': row['nh4']['step_recall'] = .49
    assert not tradeoff_gate(missed, 'candidate')['passed']
    json.dumps(tradeoff_gate(missed, 'candidate'), allow_nan=False)
    print('PASS conditional denominators, loss directions, missing classes, dual update, full-horizon gradients, recall gate')


if __name__ == '__main__':
    main()
