"""Verify carry-in handling and optimal onset matching on independent cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from event_timing import onset_counts

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
np.testing.assert_array_equal(onset_counts([1,1,0,1],[1,1,0,1],True,0),[1,1,1,0])
np.testing.assert_array_equal(onset_counts([0,0,1,0,1,0],[1,0,1,0,0,0],False,2),[2,2,2,4])
np.testing.assert_array_equal(onset_counts([0,1,0],[0,0,1],False,0),[1,1,0,0])
np.testing.assert_array_equal(onset_counts([0,1,0],[0,0,1],False,1),[1,1,1,1])
np.testing.assert_array_equal(onset_counts([0,0],[0,0],False,4),[0,0,0,0])
a.out.write_text(json.dumps({'carried_in_events_excluded':True,'maximum_cardinality_matching_passed':True,
    'tolerance_and_empty_cases_passed':True,'real_policy_performance_tested':False},indent=2));print('Onset checks passed',flush=True)
