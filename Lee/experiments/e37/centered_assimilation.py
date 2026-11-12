"""Subtract a frozen inner-period residual offset before episode adaptation."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/experiments/e35'))
from assimilate import main

if __name__=='__main__':main(centered=True)
