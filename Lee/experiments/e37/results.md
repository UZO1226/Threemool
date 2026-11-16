# Lee/e37 centered innovation results

Execution commit05cfa16. Inner calibration centers are estimated separately
for each frozen member from past observations only. Future-target-scrambling
and zero-gain reconstruction checks pass. The same20 gain pairs select NH4
gain0 and DO gain .25. Inner proper score .6737785 is only slightly below
the unchanged parent's .6746007.

Development normalized MAE .86609, NH4 frequency5.8632 percent against17.3158
actual and DO1.2000 percent against1.1579 actual still fail NH4 calibration.
DO's aggregate frequency error .0421 points is small, but this alone is not
accurate event localization. Separate event/probability metrics and16-draw
sensitivity remain in summary.json. Zero members satisfy the combined gate.
Air order passes. Centering is not a sufficient fix; no development-selected
NH4 gain is substituted for the training-selected zero. No new policy or
final test evaluation. The goal continues with conditional transition
quantiles in e38 and a nonlinear saturating reaction model in e39.
