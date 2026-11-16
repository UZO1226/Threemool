# e66: Current-exogenous conditional innovation blocks

Prospective environment hypothesis, 2026-11-17. e61 conditioned a whole-day
innovation path on the initial state and failed the safety-prediction gates.
e59 found a large gap between observation-reset and recursive forecasts;
e60 did not demonstrate a benefit from simply shuffling innovations. Test
whether selecting a short joint innovation block by the current exogenous
context improves recursive event timing and probabilities.

Keep the frozen e40 quantile weights, alpha0.5 and clipping unchanged. At
steps0,4,8 through92, use current flow, temperature, sin and cos only. Scale
these four coordinates by their mean and population standard deviation from
all initial95 rows on the158 fitting days, with standard-deviation floor1e-6.
Rank all eligible calibration donor windows by squared standardized distance
at the window start, tie by chronological donor day then start index.
Use exactly32 nearest windows, sorted by that original index before uniform
sampling. Copy a contiguous four-step joint five-channel PIT block from the
chosen window; use only three entries in the final block. NO3 stays0.5.
All starts0 through91 are eligible, so no block crosses a date boundary.

This uses no concentration, air command, future exogenous value or target
in donor selection. A block is selected at its beginning and retained for
one hour. Therefore common-noise air comparisons share exactly the same
noise path. The donor bank still uses calibration outcomes as a calibration
artifact. Exclude the complete query date during calibration diagnostics.
Use all30 calibration days for validation. No refitting or parameter search.

Primary is conditional32/block4. A global/block4 control uses the identical
sampler with all eligible windows, to separate conditioning from block size.
Retain frozen global/block95 as the principal environment reference and
reconstruct it with the unchanged historical path generator. New sampler
draws use a distinct e66 SHA256 namespace, uniformly selecting one eligible
window per block. A full-pool conditional setting must equal its own global
control bitwise; it does not claim bitwise equality with e38 block4 RNG.

First authenticate data, frozen bank and split, then independently reconstruct
neighbors, donor copying and all calibration paths. Test determinism,
full-pool identity and invariance of earlier blocks under future-exogenous
perturbation. No validation or test queries in this preflight. Then report
leave-one-day-out calibration and reused-validation forecasts without using
quality outcomes to retune the method. Independently reconstruct a full
95-step trajectory for each member and check common-noise air ordering.

Adoption requires the original point gate plus nonpositive upper95% endpoints
for NH4 and DO Brier differences versus e40 and DO versus persistence using
the existing10000 circular five-day date-bootstrap resamples. Also require
DO average precision above prevalence, AUC above0.5 and at least one new DO
onset matched within four steps at fixed probability threshold0.5, excluding
initial ongoing events. Preserve separate NH4/DO frequencies, onset counts,
errors and uncertainty, and8/16-draw sensitivity. A failed gate means no new
world adoption or policy training on it. If adopted, compare both e63 and
agent candidates in both environments to separate their effects.

Final test41 is excluded. Reused validation remains exploratory. Do not
interpret predictive or simulator gains as a physical plant intervention.
This experiment is independent of e65's fixed-world agent comparison.
