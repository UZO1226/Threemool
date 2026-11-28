# Conditional short-block result

Completed evaluation and independent audit reject the primary candidate.
All seven prospectively fixed adoption conditions fail. No new environment
is adopted and no policy is trained on this candidate. Existing e42/e63
reference results remain unchanged, and final test41 is not evaluated.

| Validation measure | Conditional32/block4 | Global/block4 control | Observed |
| --- | --- | --- | --- |
| NH4 violation frequency, percent | 4.996053 | 13.850000 | 17.315789 |
| DO violation frequency, percent | .373684 | .234211 | 1.157895 |
| NH4 Brier | .150552 | .140256 | |
| DO Brier | .013208 | .012178 | |

The primary normalized MAE is.720521 and NH4 MAE1.849906. Better average
concentration errors do not overcome the12.319737-point NH4 frequency error,
the DO calibration condition relative to the frozen reference, or zero of
five members meeting both frequency bounds. Both common-noise recursive
air-order discrepancies are exactly zero.

NH4 Brier is worse than frozen e40: paired-date difference+.024066, circular
five-day bootstrap95% interval[.008913,.038373]. DO versus e40 has difference
-.000119, interval[-.000837,.000420], which does not prove noninferiority
under the declared upper-endpoint rule. DO is worse than daily initial-state
persistence, difference+.001629, interval[.00000954,.00392039]. It is also
worse than the identical-sampler global/block4 control: difference+.001031,
interval[.00000329,.00253060]. Preserve the full intervals in audit.json.

DO AP equals prevalence.011579 and AUC is.489750. New onset matches are0/25
for DO and0/89 for NH4 at tolerances0,1 and4 steps under the fixed threshold
0.5, excluding initial ongoing events. There is no demonstrated repair of
onset timing or probability calibration. Sixteen-draw sensitivity predicts
NH4 5.226645% and DO .387829%, retaining the large NH4 underprediction.

Calibration-only leave-one-date-out results had NH4 predicted4.990351%
versus5.508772% observed, but DO2.211404% versus10.736842% observed. These
are not validation outcomes or a basis for retuning. Conditional selection
on current flow, temperature and clock does not demonstrate transferable
innovation calibration across these dates. It does not isolate which
unobserved process, model error or distribution shift caused the failure.
The global short-block control is a mechanism contrast, not an adopted
fallback. Do not promote a favorable individual measure into a new gate.

All calibration pools and1200 joint paths independently reconstruct. The
final audit independently rebuilds every validation donor and path for both
conditions, all aggregate scores and air-order checks. A complete95-step
sklearn/scalar-interpolation trajectory for every member on the first date
and draw agrees within2.22e-16 for the global control and1.11e-16 for the
primary. Frozen, original-model and persistence reference event metrics
reproduce the prior audit. This establishes the specified implementation
checks, not physical plant safety or independent out-of-sample confirmation.

artifacts.json records the completed archive in the existing environment
draft, with remotely verified size and SHA256. It contains compact outputs
and provenance, excludes all observations and PIT arrays, and depends on
the already archived e40 weights and the authenticated local DARROW data.
The audit's summary digest canonicalizes JSON line endings; archive entries
retain and verify their exact stored-byte digests. No weights were fitted.
