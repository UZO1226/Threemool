# Lee/e34 context-consistent teacher-forcing results

Execution commit115a856. The independent initial-step, target, fixed-context
and gradient checks pass. Five80-epoch fits complete. Inner selection chooses
epoch80,n1_d1, proper score .6244246 versus e32 .6412332 on the same inner dates.
This improves inner selection but still does not validate safety forecasts.

Development normalized MAE is .76011 and NH4 MAE1.45093mg/L. NH4 violation9.1053
percent underpredicts17.3158 actual by8.2105 points. DO1.4316 percent differs
from1.1579 actual by .2737 points. Only two members meet both safety tolerances;
four are required. Common-noise recursive air-order checks pass, but the
unchanged adoption gate fails NH4 calibration and member coverage.

Independent full-day Torch/NumPy and selected-score reconstruction pass. NH4
bias date-block95-percent interval[-11.01,-5.11] points remains negative. Full
probability, timing and horizon diagnostics are in audit.json. No policy or
official test evaluation occurs. e36 freezes these weights to test whether
proper-score bias calibration changes the outcome, not to claim an agent gain.
