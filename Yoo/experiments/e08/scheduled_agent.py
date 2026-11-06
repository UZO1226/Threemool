"""미리 정한 갱신 횟수 뒤 학습된 FCSRL 표현을 고정하는 대조군."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/wwtp'))
from ppo_fcsrl import PPOFCSRLAgent

REPRESENTATION_MODULES = ('encoder', 'fixed_encoder', 'feasi_head', 'proj_layer', 'post_proj')


def representation_digest(agent):
    h = hashlib.sha256()
    for name in REPRESENTATION_MODULES:
        for key, value in getattr(agent, name).state_dict().items():
            h.update(f'{name}.{key}'.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ScheduledAgent(PPOFCSRLAgent):
    def __init__(self, *args, freeze_after=0, **kwargs):
        if type(freeze_after) is not int or freeze_after < 0:
            raise ValueError('freeze_after must be a nonnegative integer; zero disables freezing')
        super().__init__(*args, **kwargs)
        if self.repr_config.frozen:
            raise ValueError('This control must start with a trainable representation')
        self.freeze_after = freeze_after
        self.updates_done = 0
        self.frozen_digest = None

    def update(self, buf):
        info = super().update(buf)
        self.updates_done += 1
        if self.freeze_after and self.updates_done == self.freeze_after:
            # 지정된 갱신의 원본 표현 학습과 EMA까지 마친 뒤 다음 갱신부터 고정한다.
            self.repr_config = replace(self.repr_config, frozen=True)
            for name in REPRESENTATION_MODULES:
                getattr(self, name).requires_grad_(False)
            self.frozen_digest = representation_digest(self)
        if self.frozen_digest is not None:
            if representation_digest(self) != self.frozen_digest:
                raise RuntimeError('Representation changed after its fixed freeze boundary')
        info['representation_frozen_next'] = int(self.repr_config.frozen)
        return info

    def save(self, path):
        super().save(path)
        (Path(path) / 'freeze_schedule.json').write_text(json.dumps({
            'freeze_after': self.freeze_after, 'updates_done': self.updates_done,
            'frozen_digest': self.frozen_digest,
            'note': 'Inference checkpoint only; optimizer and collector resume are not implemented.'}, indent=2))
