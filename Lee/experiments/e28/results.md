# Lee/e28 season-spanning partition results

Execution commita3fb76e. The timestamp-only158/30 partition passes count,
disjointness and official cutoff checks. Original and new fitting sets have
the same minimum temperature14.0183 degrees C, so the experiment does not
expand the coldest range. Model normalization is fit-only; evaluation/calibration
scoring SD is frozen to the original reference, making reported MAEs comparable.

Inner selection chooses epoch80,n1_d1. Development normalized MAE .80463 and
NH4 MAE1.34410 improve relative to e24, but NH4 violation prediction is7.9664
percent against17.3158 actual. DO prediction2.1487 percent against1.1579 actual
has error .9908 points, worse than the original GBM's .6421-point error.
The unchanged gate fails for both NH4 and DO calibration and has zero calibrated
members. The lower inner score cannot be compared directly with e24's inner
score as evidence of improvement because the inner dates differ.

Five-day date-block95-percent intervals give NH4 bias[-11.90,-6.64] points and
DO bias[.170,1.901]. Full-day Torch/NumPy agreement and saved-score reconstruction
pass. Separate probability, event and horizon diagnostics appear in audit.json.
No agent training, test evaluation or official protocol changes. This partition
alone is not a sufficient fix and is not combined with e29's continuation.
