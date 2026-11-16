"""Choose an action offset using air consumption alone, without quality scores."""
import math

MAX_VOLUME=9000*.25*95


def match_air(air_only,target,native_air,max_queries=25,relative_tolerance=.001):
    if not 0<=target<=MAX_VOLUME:raise ValueError('Target outside attainable action endpoints')
    if not 0<=native_air<=MAX_VOLUME:raise ValueError('Invalid native air')
    points={-2.:0.,0.:float(native_air),2.:MAX_VOLUME}
    trace=[{'offset':x,'air_m3':y,'source':'native' if x==0 else 'analytic clipped endpoint'} for x,y in points.items()]
    def rank(x):return (abs(points[x]-target),abs(x),x)
    best=min(points,key=rank);queries=0
    strict_tolerance=max(1.,abs(target)*relative_tolerance)
    if abs(points[best]-target)>strict_tolerance:
        pairs=[(-2.,0.),(0.,2.)]
        brackets=[(left,right) for left,right in pairs if (points[left]-target)*(points[right]-target)<=0]
        left,right=min(brackets,key=lambda pair:(abs(pair[0])+abs(pair[1]),pair))
        while queries<max_queries and abs(points[best]-target)>strict_tolerance:
            middle=(left+right)/2
            air=float(air_only(middle));queries+=1
            if not math.isfinite(air) or not -1e-6<=air<=MAX_VOLUME+1e-6:raise ValueError('Invalid air callback output')
            points[middle]=air;trace.append({'offset':middle,'air_m3':air,'source':'air-only callback'})
            best=min(points,key=rank)
            if (air-target)*(points[left]-target)>0:left=middle
            else:right=middle
    error=points[best]-target
    return {'offset':best,'target_air_m3':float(target),'achieved_air_m3':points[best],
        'air_error_m3':error,'relative_air_error':error/target if target>0 else 0.,
        'matched_within_one_percent':abs(error)<=abs(target)*.01 if target>0 else abs(error)<1e-9,
        'strict_tolerance_reached':abs(error)<=strict_tolerance,'callback_queries':queries,'trace':trace}
