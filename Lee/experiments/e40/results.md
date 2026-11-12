# Result

Source commit d31b068. Inner selection chooses pit_b95_a0.5. The corrected
quantile model passes every unchanged e16 point-estimate adoption gate, with
four of five members calibrated. The16-draw sensitivity keeps ensemble NH4
and DO frequency errors within the same tolerances. No final test was used.

| Metric | Original delta GBM | e40 |
| --- | --- | --- |
| Normalized MAE | 1.026065 | .775476 |
| NH4 MAE | 2.482256 | 1.977722 |
| NH4 frequency percent | 28.4842 | 13.4993 |
| DO frequency percent | 1.8000 | .7211 |
| Observed NH4 frequency percent | 17.3158 | 17.3158 |
| Observed DO frequency percent | 1.1579 | 1.1579 |
| NH4 Brier | .217684 | .126486 |
| DO Brier | .021074 | .013327 |

Both recursive action-order checks pass at numerical precision. The fast
predictor equals sklearn exactly, and independent full-day sklearn inference
with NumPy interpolation reconstructs the selected stochastic forecast.
Weights and the PIT bank are frozen in e42/environment_freeze.json before
controller training. No e41 long-history fitting is needed at this stage;
retain that prepared alternative without presenting its checks as a result.

## Scope and uncertainty

These are exploratory reused validation dates. Date-block95 percent intervals
for NH4 frequency bias are[-7.48755,.09079] percentage points and DO bias
[-1.60594,.90725]. Point-estimate gates passing does not prove population bias
lies within tolerance. The proposed policy comparison is within a frozen
surrogate, not real-plant causal confirmation.

Event timing remains weak. NH4 probability-at-half recall is.18997 and precision
.75301; DO recall and precision are zero at that threshold. Its DO Brier improves
over the old model but is worse than the all-safe persistence baseline. The
observed DO events are rare,44 of3800 steps. NH4 predicted runs are longer and
less frequent than observed. Quantile intervals are not claimed to have nominal
90 percent coverage; NO3 is a deterministic median by design. See summary.json
and audit.json for complete reliability, horizon, member and event diagnostics.

A separate fixed-model event audit compares ranking and explicit new onsets
against both original and persistence references. NH4 average precision rises
from.30003 to.42675 and ROC AUC from.67821 to.71631, while recall at probability
.5 drops from.53191 to.18997. Within one hour, only3 of89 true new NH4 onsets
are matched versus10 for the original model. Ongoing initial violations are
excluded and matching is one-to-one; no diagnostic threshold was tuned.

DO average precision is.01149 versus.01452 original and prevalence.01158;
ROC AUC.48475 is near chance. No new DO onset is matched by either model.
The date-paired Brier difference intervals are[-.13744,-.05248] for NH4 and
[-.01451,-.00203] for DO versus the original, but DO is worse than persistence,
with difference interval[.000046,.004223]. These reused-date diagnostics
confirm an improvement in probability error relative to the original while
contradicting any claim that event onset prediction or DO ranking is solved.
See event_reference_audit.json and onset_check.json. The frozen policy study
must retain this model limitation and cannot establish real-plant safety.

This is the first tested environment candidate to satisfy the predefined
exploratory gates, not evidence that an agent extension beats FCSRL. Baseline
and extension must now use this exact same environment, cost definition and
budget, with separate matched-air seed-paired comparisons. Preserve the final
test41 for the team's agreed comparison.
