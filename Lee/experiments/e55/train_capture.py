"""Re-run the original policy with a passive, separately stored training stream."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

from capture_env import CaptureEnv
from quantile_env import frozen_digest
import train_policy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--capture-dir', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    assert '--environment' in remaining and remaining[remaining.index('--environment') + 1] == 'new'
    assert '--method' in remaining and remaining[remaining.index('--method') + 1] == 'ppolag_fcsrl'
    if args.capture_dir.exists() and any(args.capture_dir.iterdir()):
        raise ValueError(f'Refusing to overwrite capture: {args.capture_dir}')
    checks = json.loads(Path(__file__).with_name('capture_check.json').read_text())
    assert checks['passed']
    for name, expected in checks['runtime_sources'].items():
        assert frozen_digest(Path(__file__).with_name(name)) == expected
    class RecordingEnv(CaptureEnv):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs, capture_path=args.capture_dir)
    original_train = train_policy.train
    def captured_train(*positional, **kwargs):
        kwargs['run_metadata'] = dict(kwargs['run_metadata'])
        kwargs['run_metadata']['passive_capture'] = {
            'directory': str(args.capture_dir.resolve()),
            'extra_environment_steps': 0,
            'sources': {p.name: frozen_digest(p) for p in Path(__file__).parent.glob('*.py')},
        }
        return original_train(*positional, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'argv', [sys.argv[0], *remaining]))
        stack.enter_context(patch.object(train_policy, 'FrozenQuantileEnv', RecordingEnv))
        stack.enter_context(patch.object(train_policy, 'train', captured_train))
        train_policy.main()


if __name__ == '__main__':
    main()
