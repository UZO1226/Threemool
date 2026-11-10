"""Kaggle 원 설명과 공정도에 따른 DO 센서와 직접 폭기 구간의 대응."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'e30'))
from surrogate_dw import STATE, EXO, ZONES, monotone_constraints as legacy_constraints

DO_AIR = {'do1': 'air3', 'do2': 'air4', 'do3': 'air5'}
FEATURES = STATE + EXO + ['current_' + z for z in ZONES] + ['previous_' + z for z in ZONES]


def monotone_constraints(target):
    name = STATE[target]
    if name not in DO_AIR:
        return legacy_constraints(target)
    result = np.zeros(len(FEATURES), dtype=int)
    for prefix in ('current_', 'previous_'):
        result[FEATURES.index(prefix + DO_AIR[name])] = 1
    return result


def verify():
    # Expected positions are explicitly checked against the source column layout.
    for target, expected in ((2, [16, 21]), (3, [17, 22]), (4, [18, 23])):
        c = monotone_constraints(target)
        assert np.flatnonzero(c).tolist() == expected
        assert np.all(c[expected] == 1)
        assert not np.array_equal(c, legacy_constraints(target))
    for target in (0, 1):
        assert np.array_equal(monotone_constraints(target), legacy_constraints(target))
