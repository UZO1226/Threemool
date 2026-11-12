"""Train-only risk-stratified episode sampling without changing reward or cost."""
import numpy as np


def risk_day_weights(table, days):
    risk = np.array([float((table.loc[day].nh4.to_numpy()[1:] > 4).mean()) for day in days])
    count = max(1, int(np.ceil(len(days) * .25)))
    high = np.argsort(-risk, kind='stable')[:count]
    mask = np.zeros(len(days), dtype=bool)
    mask[high] = True
    weights = np.zeros(len(days))
    weights[mask] = .5 / mask.sum()
    weights[~mask] = .5 / (~mask).sum()
    return weights, {'top_fraction': .25, 'sampling_probability': .5,
                     'risk': 'observed training-day NH4 violation fraction',
                     'high_risk_days': [days[i] for i in high],
                     'weights': weights.tolist()}
