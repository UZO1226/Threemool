"""Repeat the distributional family with its nominal constraints respected."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e38'))
from quantile_dynamics import main
from monotone_bounds import preserve_bounds

if __name__=='__main__':main(training_context=preserve_bounds,strict_order=True)
