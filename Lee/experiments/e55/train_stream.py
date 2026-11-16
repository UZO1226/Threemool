"""Batch-runner entry point, storing generated captures beside each policy."""
import argparse
from pathlib import Path
import sys

from train_capture import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    known, _ = parser.parse_known_args()
    sys.argv.extend(['--capture-dir', str(known.out / 'transitions')])
    main()
