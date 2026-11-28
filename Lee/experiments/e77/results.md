# Conditional errors explain why pooled accuracy is insufficient

Executed at453cd30 on2026-11-29. All five e76 summaries reconstruct from
authenticated captures. diagnosis.json preserves calibration and exposed
held results, fixed strata and all model/persistence alarm disagreements.

| Seed | Held current DO below1 share | Coverage at current DO1 to1.5 | Coverage at current DO at least2 | Misses at current DO at least2 |
| --- | --- | --- | --- | --- |
| 0 | .8518 | .8473 | .9244 | 10 |
| 1 | .9029 | .8034 | .5491 | 22 |
| 2 | .8361 | .8359 | .8862 | 13 |
| 3 | .8657 | .8222 | .9071 | 11 |
| 4 | .8691 | .8063 | .8758 | 10 |

No observed next-DO violations occur below current DO1 in these calibration
or held blocks. This dominant easy stratum masks poor conditional coverage.
In seed1,22 of24 held misses occur when current DO is already at least2.
Calibration shows the same failure direction: upper coverage in that stratum
is .6897, with8 missed events out of66. A global .95 quantile does not imply
.95 coverage in every state region, and MSE reduction is a different target.
These observations establish association, not a causal decomposition.

This is an aggregate diagnostic with no new fitting, controller evaluation,
environment query or final test access. It does not show policy improvement.
The source models and audit are bound by SHA256 in diagnosis.json and remain
preserved in the e76 private artifact. All new output consists of aggregate
JSON committed here, without observation rows or raw transition samples.

Next hypothesis is separately specified as e78: fixed current-DO conditional
upper margins with a persistence alarm floor. This is a changed forecasting
algorithm, not a relaxed coverage or violation-miss admission gate. Wider
bounds can increase false positives and reduce useful controller freedom;
these must be reported and matched-air policy gains remain unproven.
