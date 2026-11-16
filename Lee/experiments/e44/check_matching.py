"""Check smooth, nonmonotone, saturated and unattainable matching cases."""
import argparse
import json
import math
from pathlib import Path

from air_match import match_air,MAX_VOLUME

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
cases={}
functions={'linear':lambda x:(x+2)/4*MAX_VOLUME,
    'curved':lambda x:((x+2)/4)**2*MAX_VOLUME,
    'nonmonotone':lambda x:max(0,min(MAX_VOLUME,((x+2)/4+.04*math.sin(4*math.pi*x))*MAX_VOLUME))}
for name,function in functions.items():
    seen=[]
    def callback(x):assert -2<=x<=2;seen.append(x);return function(x)
    result=match_air(callback,MAX_VOLUME*.63,function(0))
    assert result['strict_tolerance_reached'] and result['matched_within_one_percent']
    assert len(seen)==result['callback_queries']<=25
    assert abs(function(result['offset'])-MAX_VOLUME*.63)<=MAX_VOLUME*.63*.001
    cases[name]={'offset':result['offset'],'relative_air_error':result['relative_air_error'],'queries':len(seen)}
for target,offset in [(0,-2),(MAX_VOLUME,2)]:
    result=match_air(lambda x:(_ for _ in ()).throw(AssertionError('Unneeded callback')),target,MAX_VOLUME*.4)
    assert result['offset']==offset and result['callback_queries']==0
failure=match_air(lambda x:0 if x<.2 else MAX_VOLUME,MAX_VOLUME*.5,0)
assert not failure['matched_within_one_percent'] and failure['callback_queries']==25
small=match_air(functions['linear'],.5,MAX_VOLUME*.4)
assert not small['matched_within_one_percent']
try:match_air(lambda x:float('nan'),MAX_VOLUME*.4,0)
except ValueError:pass
else:raise AssertionError('Nonfinite air accepted')
result={'quality_scores_are_not_inputs':True,'smooth_and_nonmonotone_cases':cases,
    'analytic_saturation_endpoints_passed':True,'unattainable_discontinuity_reported':True,
    'query_budget_enforced':True,'nonfinite_callback_rejected':True,'policy_performance_tested':False}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
