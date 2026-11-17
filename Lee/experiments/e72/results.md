# Held component-risk trade diagnostic

The complete declared diagnostic passes for every actual policy seed0 through9.
The four current-regime counts reconstruct all85120 held rows and every
aggregate counter per seed. All aggregate counts agree with the retained
first implementation. Existing e71 changed-action and summed-risk-worse
counts reconstruct exactly.1024 original-sklearn probes pass per seed.
No new environment query, model fitting or candidate evaluation is performed.

| Seed | Predicted NH4 increases | Predicted DO increases | NH4 increases while DO decreases | Componentwise-admissible gain available |
| --- | --- | --- | --- | --- |
| 0 | 961 | 1320 | 847 | 14282 |
| 1 | 1345 | 4434 | 1345 | 17825 |
| 2 | 1456 | 1587 | 1455 | 16503 |
| 3 | 5475 | 4536 | 2444 | 36294 |
| 4 | 4107 | 3601 | 2192 | 18116 |
| 5 | 1356 | 1942 | 1356 | 18199 |
| 6 | 2355 | 2691 | 2355 | 18312 |
| 7 | 4758 | 1395 | 3340 | 29136 |
| 8 | 1075 | 3754 | 934 | 15989 |
| 9 | 3339 | 2139 | 3212 | 21402 |

The first three columns describe the fitted e71 native action relative to
same-state e63 native under the fixed horizon4 risk heads. The last column
counts states where some of the original nine proposals improves the summed
rate without increasing either component beyond1e-12. It is a feasibility
screen, not the action chosen by a new calibrated controller or measured
water-quality improvement. Both higher-air and lower-air opportunities occur.

Current regimes are0 neither violation,1 NH4 only,2 DO only and3 both,
using current NH4>4 and max of DO1/DO2/DO3>2. For example, original seeds
have5906 current-DO-only held rows,626 with NH4 prediction increasing and
only13 with DO prediction increasing. Exposed fresh seeds have8478 such
rows,1350 with NH4 prediction increasing and0 with DO prediction increasing.
These descriptive counts are not independent samples, future-event outcomes,
or a causal explanation of the failed fresh quality result.

The findings motivate e73's separately declared componentwise exclusion,
but do not prove it will improve actual-air-matched quality. Recursive state
shift, risk-head misspecification and the post-wrapper air offset remain.
Previously fresh seeds5 through9 are now exposed development data. No final
test41 data is used and reused validation40 remains exploratory.

The first run at8a2c503 reconstructs aggregate quantities but omits the
declared regime grouping. Its records are retained, not called a completed
full diagnostic. The final run atf9eac33 adds grouping, pins its source and
requires exact initial-count identity. summary.json binds both source and
previous-summary hashes. Reproduce the two stages in that order or run the
completed implementation without --previous into a new output directory.

All initial and final diagnostic records, referenced e71 model archive indexes
and exact provenance are preserved in the existing private draft443333191,
asset757658153, lee-e72-component-risk-diagnostic.zip,186822 bytes,
SHA256 3e854f02dcae78977fe92136030f05e10e1dd98095f5471d2d698cdd19383413.
artifacts.json records per-file hashes and verified remote size/digest.
No observations, capture arrays or stochastic banks are included. Original
NN and risk weights are referenced from both complete e71 archives.
