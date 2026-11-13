"""Expand public-observation inputs while retaining the original initial policy."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/code/darrow'))
sys.path.insert(0, str(ROOT.parent / 'FCSRL'))
import fcsrl_compat
from ppo_fcsrl import PPOFCSRLAgent
from ppo_lag import RunningNorm

OBSERVATION_DIM = 17
LAG_COLUMNS = [0, 1, 2, 3, 4, 10]
MODES = ('history', 'repeated')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


class ExpandedLinear(nn.Linear):
    def __init__(self, original, insert_at=11):
        assert original.weight.device.type == 'cpu'
        with torch.random.fork_rng(devices=[]):
            super().__init__(original.in_features + 6, original.out_features,
                             bias=original.bias is not None, device=original.weight.device,
                             dtype=original.weight.dtype)
        common = torch.cat([torch.arange(insert_at), torch.arange(insert_at + 6, self.in_features)])
        extra = torch.arange(insert_at, insert_at + 6)
        self.register_buffer('common_columns', common, persistent=False)
        self.register_buffer('extra_columns', extra, persistent=False)
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, common] = original.weight
            if self.bias is not None:
                self.bias.copy_(original.bias)
        self.weight.requires_grad_(original.weight.requires_grad)
        if self.bias is not None:
            self.bias.requires_grad_(original.bias.requires_grad)
        self.train(original.training)

    def forward(self, value):
        common = F.linear(value.index_select(-1, self.common_columns),
                          self.weight.index_select(1, self.common_columns), self.bias)
        extra = F.linear(value.index_select(-1, self.extra_columns),
                         self.weight.index_select(1, self.extra_columns), None)
        return common + extra


class HistoryFCSRL(PPOFCSRLAgent):
    def __init__(self, s_dim=17, a_dim=1, device='cpu', config=None, repr_config=None, mode='history'):
        assert s_dim == 17 and mode in MODES
        super().__init__(11, a_dim, device, config=config, repr_config=repr_config)
        self.history_mode = mode
        self.encoder.zs_net.layers[0] = ExpandedLinear(self.encoder.zs_net.layers[0])
        self.fixed_encoder.zs_net.layers[0] = ExpandedLinear(self.fixed_encoder.zs_net.layers[0])
        for module in (self.pi, self.vr, self.vc):
            module[0] = ExpandedLinear(module[0])
        self.obs_normalizer = RunningNorm(17)
        params = [p for module in (self.encoder, self.feasi_head, self.proj_layer, self.post_proj)
                  for p in module.parameters()]
        self.encoder_optim = torch.optim.Adam(params, lr=self.repr_config.encoder_lr)
        self.opt_pi = torch.optim.Adam(list(self.pi.parameters()) + [self.log_std], lr=self.config.actor_lr)
        self.opt_v = torch.optim.Adam(list(self.vr.parameters()) + list(self.vc.parameters()), lr=self.config.critic_lr)
        assert not any(p.requires_grad for p in self.fixed_encoder.parameters())

    def save(self, path):
        super().save(path)
        paths = [Path(__file__), Path(__file__).with_name('history_env.py')]
        metadata = {'mode': self.history_mode, 'current_dimension': 11, 'expanded_dimension': 17,
                    'previous_columns': LAG_COLUMNS, 'initial_extra_columns_zero': True,
                    'runtime_sources': {p.name: digest(p) for p in paths}}
        (Path(path) / 'observation_history.json').write_text(json.dumps(metadata, indent=2))

    def load(self, path):
        metadata = json.loads((Path(path) / 'observation_history.json').read_text())
        assert metadata['mode'] == self.history_mode
        assert metadata['current_dimension'] == 11 and metadata['expanded_dimension'] == 17
        assert metadata['previous_columns'] == LAG_COLUMNS
        for name, expected in metadata['runtime_sources'].items():
            assert digest(Path(__file__).with_name(name)) == expected
        super().load(path)
        assert len(self.obs_normalizer.mean) == 17
