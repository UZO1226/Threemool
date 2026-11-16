"""Causal fixed-context donor pools and joint four-step innovation blocks."""
import hashlib

import numpy as np

BLOCK = 4
NEIGHBORS = 32
CONTEXT = ['q_in', 'temp', 'sin', 'cos']


def fit_scale(values):
    values = np.asarray(values, float)
    if values.shape != (158, 95, 4) or not np.isfinite(values).all():
        raise ValueError('Expected158 fitting days,95 current exogenous rows each')
    flattened = values.reshape(-1, 4)
    sd = flattened.std(0)
    return flattened.mean(0), np.maximum(sd, 1e-6), sd < 1e-6


def donor_pools(query, donor, query_days, donor_days, mean, sd, k=NEIGHBORS, exclude_self=True):
    query, donor = np.asarray(query, float), np.asarray(donor, float)
    if query.shape != (len(query_days), 95, 4) or donor.shape != (len(donor_days), 95, 4):
        raise ValueError('Expected95-step context for each date')
    if not np.isfinite(query).all() or not np.isfinite(donor).all() or np.any(sd < 1e-6):
        raise ValueError('Invalid context or scale')
    if len(set(donor_days)) != len(donor_days):
        raise ValueError('Donor dates must be unique')
    starts = np.arange(96 - BLOCK)
    donor_indices = (np.arange(len(donor))[:, None] * 95 + starts).reshape(-1)
    donor_context = donor[:, starts].reshape(-1, 4)
    result = []
    for index, day in enumerate(query_days):
        eligible = np.array([j for j, flat in enumerate(donor_indices)
                             if not exclude_self or donor_days[flat // 95] != day])
        if len(eligible) == 0 or (k is not None and not 1 <= k <= len(eligible)):
            raise ValueError('Insufficient eligible donor windows')
        row = []
        for step in range(0, 95, BLOCK):
            if k is None:
                selected = eligible
            else:
                distance = np.square((donor_context[eligible] - query[index, step]) / sd).sum(-1)
                selected = eligible[np.lexsort((donor_indices[eligible], distance))[:k]]
            row.append(np.sort(donor_indices[selected]))
        result.append(row)
    return result


def paths(bank, days, pools, draws=8, return_donors=False):
    bank = np.asarray(bank)
    if bank.ndim != 3 or bank.shape[0] != 5 or bank.shape[2] != 5 or bank.shape[1] % 95:
        raise ValueError('Expected complete-day five-member joint PIT bank')
    np.testing.assert_array_equal(bank[..., 1], np.full(bank.shape[:2], .5))
    if len(pools) != len(days) or any(len(row) != 24 for row in pools):
        raise ValueError('Expected24 donor pools per query date')
    values, chosen = [], []
    for member in range(5):
        for draw in range(draws):
            for day, row in zip(days, pools):
                seed = int.from_bytes(hashlib.sha256(f'e66:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                rng = np.random.default_rng(seed)
                donors = np.array([pool[rng.integers(0, len(pool))] for pool in row])
                if np.any(donors < 0) or np.any(donors + BLOCK > bank.shape[1]):
                    raise ValueError('Donor outside bank')
                if np.any(donors // 95 != (donors + BLOCK - 1) // 95):
                    raise ValueError('Block crosses a date boundary')
                indices = (donors[:, None] + np.arange(BLOCK)).reshape(-1)[:95]
                values.append(bank[member, indices].copy())
                chosen.append(donors)
    output = np.stack(values)
    return (output, np.stack(chosen)) if return_donors else output
