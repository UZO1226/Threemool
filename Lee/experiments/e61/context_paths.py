"""Episode-start-only, fixed nearest-day whole-path PIT sampling."""
import hashlib

import numpy as np


def initial_context(batch):
    true, exo, _ = batch
    result = np.column_stack([true[:, 0, 0], true[:, 0, 2:5], exo[:, 0, 1], exo[:, 0, 0]])
    if result.shape != (len(true), 6) or not np.isfinite(result).all():
        raise ValueError('Expected six finite initial-context features')
    return result


def fit_scale(fitting_context):
    if fitting_context.shape != (158, 6):
        raise ValueError('Scaling requires the158 fitting-day initial contexts')
    mean = fitting_context.mean(0)
    raw_sd = fitting_context.std(0)
    return mean, np.maximum(raw_sd, 1e-6), raw_sd < 1e-6


def donor_pools(query, donors, query_days, donor_days, mean, sd, k=5, exclude_self=True):
    if query.shape != (len(query_days), 6) or donors.shape != (len(donor_days), 6):
        raise ValueError('Context/date shape mismatch')
    if len(set(donor_days)) != len(donor_days) or np.any(sd < 1e-6):
        raise ValueError('Invalid donor dates or scaling')
    pools = []
    for index, day in enumerate(query_days):
        eligible = np.array([j for j, donor in enumerate(donor_days) if not exclude_self or donor != day])
        if not len(eligible) or (k is not None and (k < 1 or k > len(eligible))):
            raise ValueError('Insufficient eligible donor dates')
        if k is None:
            selected = eligible
        else:
            distance = np.square((donors[eligible] - mean) / sd - (query[index] - mean) / sd).sum(1)
            order = np.lexsort((eligible, distance))
            selected = np.sort(eligible[order[:k]])
        pools.append(selected)
    return pools


def paths(bank, query_days, pools, draws=8):
    if bank.ndim != 3 or bank.shape[0] != 5 or bank.shape[2] != 5 or bank.shape[1] % 95:
        raise ValueError('Expected five-member complete-day PIT bank')
    if len(pools) != len(query_days):
        raise ValueError('Pool/date shape mismatch')
    np.testing.assert_array_equal(bank[..., 1], np.full(bank.shape[:2], .5))
    output = []
    for member in range(5):
        for draw in range(draws):
            for day, pool in zip(query_days, pools):
                if np.any(pool < 0) or np.any(pool >= bank.shape[1] // 95):
                    raise ValueError('Donor index outside PIT bank')
                seed = int.from_bytes(hashlib.sha256(f'e38:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                rng = np.random.default_rng(seed)
                donor = pool[rng.integers(0, len(pool), size=1)[0]]
                # Preserve the original block95 RNG call, including its zero offset.
                offset = rng.integers(0, 1, size=1)[0]
                output.append(bank[member, donor * 95 + offset + np.arange(95)].copy())
    return np.stack(output)
