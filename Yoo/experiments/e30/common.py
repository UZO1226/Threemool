"""DARROW 전용 설정. e29의 알고리즘만 재사용하고 이전 가중치는 읽지 않는다."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
YOO = HERE.parents[1]
sys.path.insert(0, str(YOO / 'code/wwtp'))
sys.path.insert(0, str(HERE.parent / 'e29'))
from ablation import make_agent as ablation_agent
from behavior import make_agent as prior_agent
from ppo_config import PPOConfig
sys.path.insert(0, str(HERE))

METHODS = ('ppo', 'fcsrl', 'fcsrl_actor', 'fcsrl_critic', 'fcsrl_preserve', 'fcsrl_extra_ppo')


def config(updates=66):
    return PPOConfig(total_steps=updates * 6080, n_env=8, horizon=760,
                     n_chunks=1, init_log_std=-1.6, cost_limit=12.)


def make_agent(method, seed, prepared, c):
    assert method in METHODS
    return (prior_agent if method == 'ppo' else ablation_agent)(method, seed, prepared, c)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)
