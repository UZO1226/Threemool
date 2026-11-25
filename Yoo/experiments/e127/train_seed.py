"""Train the original e113 policy pair on a new training seed."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e113"))
import run as e113

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("arm", choices=e113.arms)
    p.add_argument("seed", type=int, choices=range(5, 10))
    args = p.parse_args()
    e113.train(args.out, args.arm, args.seed, 66)
