# Lee/e16: Safety calibration and recursive action-response audit

## Scope and frozen sequence

Continue e14/e15 under the explicitly requested goal. This is a personal
model-development experiment, not an amendment to Compare/PROTOCOL v1.1.
Branch lee/darrow-safety-calibration starts from main a6c2480 and incorporates
the prior Lee audit branch. Other members' files and official test outcomes
are not changed. No test observations are selected for evaluation.

Existing delta GBM and e14 direct 7 are references. A new cooperative direct
GBM excludes NO3 feedback into NH4/DO and imposes an orthant-monotone update:
next NH4 increases with current NH4, decreases with current DO and air;
next DO increases with current DO and air, decreases with current NH4.
Every DO-to-DO dependence is nonnegative. Thus, with identical initial state
and exogenous inputs, a pointwise increase in air cannot increase NH4 or
decrease any DO along the recursive trajectory. This is an explicit prior,
not a causal identification result. No unsupported mapping between the five
air zones and three DO sensors is asserted. NO3 does not feed back into these
four constrained states.

Two candidate sizes, 7 and 15 leaves, use 200 iterations, rate .05, minimum
leaf 40, L2 1, five members, reproducible 80 percent fitting subsets. The first
158 training dates fit trees; the last 30 training dates calibrate output bias.
Compare both uncalibrated and train-calibrated variants. Calibrate a constant
per-output update bias with bounded optimization of normalized recursive MAE
and NH4/DO soft-exceedance calibration. Bias is independent of actions and
states and therefore preserves the monotonic prior. No development labels
enter fitting, calibration, or optimization. Candidate selection uses inner
calibration scores only; evaluate every candidate on the 40 development dates
for transparent comparison. These development dates have been used in prior
experiments and are not a new independent confirmation set.

## Adoption gate before agent training

For the candidate selected by inner score, development recursive normalized
MAE must be at least 10 percent below the original delta GBM, NH4 recursive
MAE at least 10 percent below it, NH4 violation-frequency absolute error at
most 5 percentage points, and DO frequency error at most 1 percentage point.
Neither safety calibration error may worsen relative to the original GBM.
At least four of five model members must satisfy NH4/DO frequency tolerances.
Report concentration and delta one-step metrics separately, a persistence
reference, day-paired uncertainty, and multi-horizon recursive errors.
No passing model means no new expensive policy training on that model.

Before adopting a model, check raw input hashes, prepared-table correspondence,
date separation, full-trajectory air-response order on every day/member, and
action support. If this model family fails, inspect the failure before choosing
the next family. Failed models and optimization results remain available.

After an environment passes, freeze it, recalibrate the budget from training
replay only, compare base FCSRL and a prospectively specified extension with
paired training seeds, and assess a matched-air reference without extrapolation.
Keep NH4/DO separate. Do not attribute environment changes to agent changes.

## Data and limitations

DARROW Kaggle files and the 188/40/41 chronological split follow v1.1. Model
learning uses only the 188 training days. Original code and data hashes are
recorded at execution. CPU packages follow e15/runtime.json. Original FCSRL
d882b91054d52c7a63a18a21f7db67bbbed82b63 remains unchanged.
Observational fit cannot validate unobserved interventions. The official
project describes a process model used for RL training, distinct from this
empirical surrogate: https://www.wastewater.ai/holistic-optimisation-of-wastewater-treatment-plants/
The Kaggle data page was consulted but yielded no readable sensor schematic:
https://www.kaggle.com/competitions/dynamic-modeling-of-wastewater-treatment-process/data
