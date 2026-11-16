# Lee/e32 persistent-context results

Execution commitabf5e75. Full95-step zero-offset equivalence to e24, RNG pairing,
cached/Torch agreement and a nonzero new-head gradient pass before training.
All five80-epoch fits finish; inner selection chooses epoch80,n1_d1, with proper
score .64123 versus e24 .67460 on the same inner dates. The additional persistent
context improves that selection score, but does not meet the adoption gate.

Development normalized MAE is .78209 and NH4 MAE1.47066mg/L. NH4 frequency8.3730
percent underpredicts17.3158 actual by8.9428 points; member predictions vary
from4.07 to14.06 percent. DO1.4257 percent is close to1.1579 actual. One member
meets both safety tolerances, not four. Sixteen draws give NH4 8.5605 percent,
so simulation sampling does not remove the failure. Air-order checks pass.

The independent full-day audit passes Torch/NumPy agreement and saved-score
reconstruction. NH4 frequency-bias date-block95-percent interval is[-12.01,-5.30]
points. DO bias interval[-.577,1.080] includes both directions. Probability,
event timing and reset metrics remain separate in audit.json. No causal
biomass identification, actual-plant improvement, policy training or test
evaluation is claimed. e33 tests an explicit kinetic temperature prior instead
of treating this result as an external blocker.
