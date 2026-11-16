"""Constraint-wise supervision with the original FCSRL representation objective.

Adapted from czp16/FCSRL d882b91054d52c7a63a18a21f7db67bbbed82b63,
Apache-2.0, PPOLagReprAgent.train_encoder. Preserve its dynamics objective,
discounted-max labels, EMA and PPO/PID application. No originality claim.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'));sys.path.insert(0,str(ROOT.parent/'FCSRL'))
import fcsrl_compat
from fcsrl.network import MLP
from fcsrl.utils import DiscDist,cosine_sim_loss
from ppo_fcsrl import PPOFCSRLAgent,feasibility_targets


class TwoHeadFCSRL(PPOFCSRLAgent):
    buffer_shapes={'part_cost':(2,),'part_next':(2,)}

    def __init__(self,*args,mode,**kwargs):
        if mode not in ('split','capacity'):raise ValueError(mode)
        super().__init__(*args,**kwargs);self.mode=mode
        r=self.repr_config
        # Restore the parent RNG so actor initialization and PPO sampling stay
        # paired. An independent second head allows the capacity control to
        # exercise its extra parameters rather than remaining an exact clone.
        seed=int(torch.initial_seed())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43000+seed)
            self.second_head=MLP(r.z_dim,[r.hidden_dim,r.hidden_dim],r.n_buckets).to(self.dev)
        self.encoder_optim.add_param_group({'params':list(self.second_head.parameters())})
        self.extra_modules=[self.second_head]

    @torch.no_grad()
    def next_parts(self,obs):
        z=self.fixed_encoder.zs(self._t(obs))
        return np.column_stack([DiscDist(head(z),self.bucket_low,self.bucket_high,self.n_buckets)
            .mean().cpu().numpy().reshape(-1) for head in (self.feasi_head,self.second_head)])

    @torch.no_grad()
    def next_feasibility(self,obs):
        parts=self.next_parts(obs)
        return parts.sum(-1) if self.mode=='split' else parts.mean(-1)

    @torch.no_grad()
    def collect_aux(self,buf,t,next_obs,cost,info):
        labels=np.column_stack([info['v_nh4'],info['v_do']])
        np.testing.assert_array_equal(labels.sum(-1),cost)
        buf['part_cost'][t]=labels if self.mode=='split' else np.repeat(np.asarray(cost)[:,None],2,axis=1)
        buf['part_next'][t]=self.next_parts(next_obs)

    def finish_buffer(self,buf):
        buf['feasi_parts']=np.stack([feasibility_targets(buf['part_cost'][...,j],buf['part_next'][...,j],
            buf['end'],buf['terminated'],self.repr_config.feasibility_discount) for j in range(2)],axis=-1)

    def update_aux(self,buf,idx):
        indices=idx.cpu().numpy()
        obs=self._t(buf['obs'][indices]);obs_next=self._t(buf['obs_next'][indices])
        action=self._t(buf['act_exec'][indices]);targets=self._t(buf['feasi_parts'][indices])
        with torch.no_grad():
            target_z=self.fixed_encoder.zs(obs_next);target_projection=self.proj_layer(target_z)
        z=self.encoder.zs(obs);pred_z=self.encoder.zsa(z,action)
        prediction=self.post_proj(self.proj_layer(pred_z))
        dynamics_loss=-cosine_sim_loss(prediction,target_projection).mean()
        losses=[];errors=[]
        for j,head in enumerate((self.feasi_head,self.second_head)):
            distribution=DiscDist(head(pred_z).unsqueeze(0),self.bucket_low,self.bucket_high,self.n_buckets)
            label=targets[:,j:j+1]
            losses.append(-distribution.log_prob(label.unsqueeze(0)).mean())
            errors.append(F.mse_loss(distribution.mean().detach().squeeze(0),label))
        feasibility_loss=torch.stack(losses).mean()
        objective=.5*dynamics_loss+feasibility_loss
        self.encoder_optim.zero_grad();objective.backward();self.encoder_optim.step()
        return {'encoder_dynamics':float(dynamics_loss.detach()),'encoder_pred_feasi':float(feasibility_loss.detach()),
            'encoder_pred_feasi_MSE':float(torch.stack(errors).mean()),'encoder':float(objective.detach()),
            'head0_feasi_loss':float(losses[0].detach()),'head1_feasi_loss':float(losses[1].detach())}

    def save(self,path):
        super().save(path);path=Path(path)
        torch.save(self.second_head.state_dict(),path/'second_head.pt')
        (path/'two_head.json').write_text(json.dumps({'mode':self.mode,'second_head_seed_offset':43000,
            'feasibility_loss':'mean of two original distribution losses','cost_limit_unchanged':True},indent=2))

    def load(self,path):
        path=Path(path);assert json.loads((path/'two_head.json').read_text())['mode']==self.mode
        super().load(path)
        self.second_head.load_state_dict(torch.load(path/'second_head.pt',map_location=self.dev,weights_only=True))
