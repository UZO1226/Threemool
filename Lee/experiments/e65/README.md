# e65: Refine the risk-only action grid

Prospective hypothesis, 2026-11-17: e63's five-point action search may miss
interior minima of the summed NH4 and DO horizon4 rate regressors. Keep its
radius0.25, policies, fitted models, inputs and training budgets fixed. Add
the missing multiples of1/32 inside the original radius, giving17 candidates.
Keep the original five first in their original order, then append positive
and negative missing offsets in increasing absolute distance. Stable argmin
therefore preserves e63 whenever no added action has strictly lower risk.
There is no new penalty, risk weight, threshold or model fit.

The primary baseline is the e64-authenticated matched e63 controller,
including its per-seed fixed offset, not its native policy or original FCSRL.
Use the unchanged quality-blind air matcher against those actual per-seed
volumes. Report native and matched results for every policy seed0 through4.
Original FCSRL is a secondary reference. Fix10000 paired-seed bootstrap
resamples, RNG2026: cost and NH4 difference interval upper endpoints must be
below zero, DO upper endpoint at most zero, and all air matches must pass.
Do not adjust the grid, radius or tie rule after observing quality outcomes.

First run a no-environment-query preflight on all85120 held training capture
states per seed. Authenticate the original captures, models and checkpoints.
Compare against e63 actions, require candidate membership and bounded radius,
and independently recompute sklearn predictions for a fixed evenly spaced
1024-row subset per seed. If the refined grid is exactly identical on every
held row for every seed, record the identity and defer an expensive study;
otherwise proceed to native smoke and full matched-air comparisons with an
independent action and trajectory audit. This preflight cannot establish
counterfactual quality, and predicted risk reduction is not a success gate.

No environment changes belong to this experiment. Model safety limitations
remain open and must be studied separately. No final test41 access. Reused
validation40 comparisons remain exploratory. A passing initial study still
requires feasible fresh policy-seed replication before the new goal closes.

Record failures and immutable source/configuration hashes. Never modify the
historical e63 runtime. Code and compact results go on the Lee branch;
observations and training captures remain local.
