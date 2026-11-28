# e77: Diagnose residual forecast misses without changing admission

Prospective diagnostic, 2026-11-29, after e76 rejects three of five seeds.
The renewed user goal is active. Earlier goal-cleared notes describe the
previous bounded task, not this renewed authorization.

Hypothesis: pooled squared error and marginal calibration are dominated by
low DO states, while rare upper-threshold transitions can remain poorly
covered. Diagnose this association without claiming a causal explanation.
Do not fit a controller, new environment, threshold or model in this study.

Reuse the frozen e76 models and authenticated official training captures.
Compare calibration updates41 through52 with exposed held updates53 through66
for all actual seeds0 through4. Keep fixed current maxDO strata
[0,1), [1,1.5), [1.5,2), [2,infinity), separately count nextDO>2 events
and safe-to-violation onsets. Report MSE, coverage, false positives and
false negatives, plus missed residual extrema and disagreement with the
existing calibrated persistence alarm. Also report residual percentiles
.50,.95,.975,.99 in calibration and held blocks, without using held
percentiles to fit a future margin. No seed exclusion or pooled success.

Binding checks require the exact e76 source, summary, audit, checkpoint
and capture hashes. Reconstruct e76 held metrics before adding diagnostics.
All output is aggregate JSON with no observations or transition samples.
The frozen e42 world and final test41 are untouched. This does not establish
on-policy generalization, iid calibration or real plant safety.

After this diagnosis, any new upper forecast or controller needs a separate
prospective specification. Existing all-five admission, matched-air and
paired-seed scientific criteria remain unchanged.
