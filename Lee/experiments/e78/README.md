# e78: Conditional upper margins with a persistence alarm floor

Prospective specification,2026-11-29, after e77 diagnoses exposed e76 errors.
Renewed active goal is to improve our own frozen-world e63 comparator at
matched air, not to pass a forecast gate alone.

Keep every e76 signed-change point model, its original source hashes and
the e42 environment fixed. Do not refit or select weights. Change only the
agent-side uncertainty estimate using calibration updates41 through52.
For fixed current maxDO strata [0,1), [1,1.5), [1.5,2), [2,infinity), compute
nonnegative .975 higher empirical quantiles of nextDO minus point forecast.
Use max(global .975 margin, stratum .975 margin), falling back to the global
margin for an empty stratum. Add this margin to the unchanged point forecast.
The final upper forecast is the maximum of that value and the original
calibrated persistence upper forecast using its unchanged .95 margin.
Stratum assignment depends on observed current DO, never a proposed action.
Thus the action monotonicity of the point forecast is preserved.

The persistence floor guarantees its alarm set is included. Consequently
nonincrease of misses versus persistence is a construction property, not
evidence that the learned model recognizes new events. Report extra alarms,
false positives, onset misses and interval width, and keep this limitation
explicit. Wider margins could prevent useful action changes and do not
constitute an improvement over FCSRL by themselves.

Evaluate all actual seeds0 through4, unchanged fit/calibration/held update
partitions, authenticated e76 dependencies and raw compiler parity. Require
all original gates: lower point MSE versus persistence and fitting constant
under paired training-date95% intervals, upper held coverage at least .95,
no more next-DO misses than original calibrated persistence, provenance and
monotonicity. Do not weaken gates, mix seeds, recalibrate from held outcomes
or modify rejected e75/e76 results. Report conditional coverage too; do not
claim conformal validity or untouched held data after repeated development.

Only if all five models pass, implement the conditional controller already
specified in e75 with e74's authenticated matched-reference geometry and
component guard, adding this upper-DO max(2, reference upper) condition.
Calibrate mean-action preservation on fitting updates1 through52 only,
with reference fallback, exact null controls and original compiled checks.
Retain the unchanged external air-only matcher and report final executed
guard breaches rather than canceling its offset. No new world is adopted.

Run all five actual seeds with native and matched results and independent
audits. Unchanged10000 paired-seed resamples with RNG2026 require cost/NH4
upper endpoints below zero, DO at most zero, and all air matches. If this
original block passes, repeat unchanged on exposed seeds5 through9 as a
separate block. If both pass, prepare actual seeds10 through14 for further
separate confirmation before adoption. No seed dropping or pooling rescue.
The official final test41 stays untouched. Preserve failures, model and
configuration provenance, extra offline computation, logs and uncertainty.
