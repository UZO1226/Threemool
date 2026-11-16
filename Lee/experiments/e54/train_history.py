"""Scoped 17-input collector adapters, preserving historical training sources."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

from history_agent import HistoryFCSRL, digest
from history_env import HistoryEnv, evaluate_history

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
import train_policy
import train_dw
from ppo_fcsrl import RepresentationConfig

METHODS = {'ppolag_fcsrl_history': 'history', 'ppolag_fcsrl_repeated': 'repeated'}


def factory(args, config):
    representation = RepresentationConfig()
    return HistoryFCSRL(17, 1, config=config, repr_config=representation, mode=METHODS[args.method]), representation


@contextmanager
def history_runtime(mode):
    class TrainingHistoryEnv(HistoryEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, mode=mode)
    def evaluator(*args, **kwargs):
        return evaluate_history(*args, **kwargs, mode=mode)
    with patch.object(train_dw, 'OBS_DIM', 17), \
         patch.object(train_policy, 'FrozenQuantileEnv', TrainingHistoryEnv), \
         patch.object(train_policy, 'evaluate', evaluator):
        yield


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method', required=True)
    parser.add_argument('--environment', required=True)
    known, _ = parser.parse_known_args()
    if known.method not in METHODS:
        train_policy.main()
        return
    assert known.environment == 'new'
    checks = json.loads(Path(__file__).with_name('preflight.json').read_text())
    assert checks['passed']
    for name, expected in checks['runtime_sources'].items():
        assert digest(Path(__file__).with_name(name)) == expected, f'Unchecked runtime source: {name}'
    with history_runtime(METHODS[known.method]):
        train_policy.main(agent_factory=factory, extra_methods=tuple(METHODS))


if __name__ == '__main__':
    main()
