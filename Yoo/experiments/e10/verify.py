"""과거값에 반응하는 모델로 연속 예측의 실측값 누출 여부를 검사한다."""
import numpy as np
import torch
from torch import nn
from history_trial import rollout, event_metrics


class HistoryProbe(nn.Module):
    def forward(self, state, exog, history):
        return state + .001 * history.mean(1)


def main():
    torch.set_num_threads(1)
    x={k:np.tile(np.arange(288,dtype=float)+1,(2,1)) for k in ('nh4','do','air','phase','temp','q_in')}
    x['phase'][:]=2
    changed={k:v.copy() for k,v in x.items()}
    changed['nh4'][:,13:]+=1000
    changed['do'][:,13:]+=1000
    model=HistoryProbe()
    pred=rollout(model,x,'history',275)
    np.testing.assert_array_equal(pred,rollout(model,changed,'history',275))
    assert not np.array_equal(pred,rollout(model,x,'current_repeat',275))
    assert not np.array_equal(rollout(model,x,'history',1),rollout(model,changed,'history',1))
    # 관측 구간 시작 전부터 지속한 위반은 새로운 시작 사건이 아니다.
    y=np.ones((1,8,2))*5
    metric=event_metrics(y,y,np.ones((1,2))*5)
    assert metric['nh4']['actual_onsets']==0 and metric['nh4']['longest_observed_run_mae_minutes']==0
    # 오른쪽에서 잘린 사건은 지속시간 MAE에서 제외한다.
    y[:,:2]=0
    metric=event_metrics(y,y,np.zeros((1,2)))
    assert metric['nh4']['matched_onsets']==1 and metric['nh4']['uncensored_matched_duration_count']==0
    print('PASS history-sensitive leakage check, reset sensitivity, left/right event censoring')


if __name__=='__main__':main()
