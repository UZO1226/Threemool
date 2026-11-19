"""FCSRL 기전 제거 대조 (Park 작성, Park/e47).

망 구조, 입력 차원, 초기값 (같은 seed), 정책 학습은 FCSRL 과 같고 표현 학습만 바꾼다.

- FCSRLRandomRepr ('fcsrl_rand'): 표현을 학습하지 않는다. 정책과 가치망은 초기값 그대로인 무작위 인코더의 z 를 입력에 붙인다
- FCSRLDynOnly ('fcsrl_dyn'): 원본 train_encoder 의 두 손실 가운데 동역학 일관성만 쓴다 (feasibility 예측 손실 계수 0).
  식과 계수 (dyn_coef 0.5) 는 원본 PPOLagReprAgent.train_encoder 와 같다

`register()` 를 부르면 agent.METHODS 에 두 이름을 더한다. 기존 'ppo', 'fcsrl' 은 바뀌지 않는다.
"""
import torch

from agent import FCSRL, METHODS
from fcsrl.utils.misc import cosine_sim_loss


class FCSRLRandomRepr(FCSRL):
    name = 'fcsrl_rand'

    def train_aux(self, buf, idx):
        return {}

    def after_update(self):
        pass


class FCSRLDynOnly(FCSRL):
    name = 'fcsrl_dyn'

    def train_aux(self, buf, idx):
        i = idx.numpy()
        obs, obs_next = self.tensor(buf['obs'][i]), self.tensor(buf['obs_next'][i])
        act = self.tensor(buf['act_exec'][i])
        with torch.no_grad():
            target_ps = self.proj_layer(self.fixed_encoder.zs(obs_next))
        pred_zs = self.encoder.zsa(self.encoder.zs(obs), act)
        pred_ps = self.post_proj(self.proj_layer(pred_zs))
        dynamics_loss = -cosine_sim_loss(pred_ps, target_ps).mean()
        loss = 0.5 * dynamics_loss
        self.encoder_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        return {'encoder_dynamics': dynamics_loss.item(), 'encoder': loss.item()}


ABLATIONS = {'fcsrl_rand': FCSRLRandomRepr, 'fcsrl_dyn': FCSRLDynOnly}


def register():
    METHODS.update(ABLATIONS)
