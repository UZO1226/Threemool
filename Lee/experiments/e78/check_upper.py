"""Check fixed stratum boundaries, signed residuals and alarm containment."""
import json
from pathlib import Path
import hashlib

import numpy as np

from upper_forecast import strata, calibrate_upper, upper_prediction


def main():
    np.testing.assert_array_equal(strata([0, .999, 1, 1.499, 1.5, 1.999, 2, 3]), [0, 0, 1, 1, 2, 2, 3, 3])
    current = np.array([.2, .3, 1.2, 1.3])
    point = current.copy()
    config = calibrate_upper(current - .1, point, current)
    assert config['global_margin'] == 0 and config['stratum_margins'] == [0.] * 4
    config = calibrate_upper(np.array([.2, 1., 1.8, 2.5]), point, current)
    assert config['stratum_counts'] == [2, 2, 0, 0]
    assert config['stratum_margins'][2:] == [config['global_margin']] * 2
    x = np.linspace(0, 3, 101)
    low = upper_prediction(x - .5, x, config)
    high = upper_prediction(x + .5, x, config)
    assert np.all(low <= high) and np.all(low >= x + config['persistence_margin'])
    target = np.linspace(0, 4, 101)
    assert np.count_nonzero((target > 2) & (low <= 2)) <= np.count_nonzero((target > 2) & (x + config['persistence_margin'] <= 2))
    source = Path(__file__).with_name('upper_forecast.py')
    print(json.dumps({'passed': True, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'checks': ['fixed_boundaries', 'nonnegative_margins', 'empty_stratum_fallback', 'action_order', 'persistence_alarm_containment']}))


if __name__ == '__main__':
    main()
