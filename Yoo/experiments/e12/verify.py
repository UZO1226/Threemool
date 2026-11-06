"""미래 단계 gradient와 위반 누락 및 거짓경보 통과 기준을 검증한다."""
import copy
import json
import torch
from torch import nn
from trial import CFG, losses, gates


class Accumulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.01))
        self.delta_sd=torch.ones(2)

    def forward(self,state,exog,extra):return state+self.weight


def main():
    torch.set_num_threads(1)
    h=CFG['horizon'];model=Accumulator()
    args=(torch.zeros(2,h+1,2),torch.zeros(2,h,6),torch.zeros(2,h,8),torch.ones(2),torch.tensor([.1,.1]))
    single,_=losses(model,*args,False);g1=torch.autograd.grad(single,model.weight)[0]
    multi,_=losses(model,*args,True);gm=torch.autograd.grad(multi,model.weight)[0]
    torch.testing.assert_close(g1,torch.tensor(.02))
    torch.testing.assert_close(gm,torch.tensor(.02*(h+1)*(2*h+1)/6))
    rows=[]
    metrics={'step_recall':.8,'step_precision':.8,'fpr':.05,'mae':1.,'onset_f1':.5,'longest_observed_run_mae_minutes':10.}
    for seed in CFG['seeds']:
        for mode in ('reference','single_mse','multi_boundary'):
            for horizon in (1,275):
                r={'seed':seed,'mode':mode,'split':'inner_holdout','reset_steps':horizon,
                   'nh4':copy.deepcopy(metrics),'do':copy.deepcopy(metrics),'cost':{'daily_mae':10.}}
                if mode=='multi_boundary':r['nh4']['step_recall']=.9
                rows.append(r)
    assert gates(rows,'inner_holdout','multi_boundary')['screen_passed']
    missed=copy.deepcopy(rows)
    for r in missed:
        if r['mode']=='multi_boundary':r['nh4']['step_recall']=.1;r['cost']['daily_mae']=0.
    assert not gates(missed,'inner_holdout','multi_boundary')['screen_passed']
    alarm=copy.deepcopy(rows)
    for r in alarm:
        if r['mode']=='multi_boundary':r['nh4']['step_recall']=1.;r['nh4']['fpr']=1.
    result=gates(alarm,'inner_holdout','multi_boundary')
    assert not result['screen_passed']
    json.dumps(result,allow_nan=False)
    print('PASS full-horizon gradients, low-recall rejection, all-alarm rejection, bool JSON')


if __name__=='__main__':main()
