# Executed-air smoothing fails despite smaller command changes

The full sourcee664aca study finishes all five fixed-gain comparisons,
independent native and matched-feedback audits and the unchanged five-seed
report. Every final base checkpoint, source/config/budget/freeze guard and
original native replay passes. No PPO learning, fitted predictor or extra
environment-training transition is added. Final test41 remains unused.

| Matched-air mean | Original FCSRL | Gain.5 air filter |
| --- | --- | --- |
| Air, m3 per23.75h | 52020.577481 | 52011.134351 |
| Summed violation cost | 22.464125 | 23.110875 |
| NH4 violation, percent | 23.208684 | 23.891579 |
| DO violation, percent | .437763 | .435658 |

All five actual air matches pass, with absolute relative errors below.091
percent. Cost difference+.646750 has paired-policy-seed95 interval
[.355875,.945375]. NH4 difference+.682895 percentage points has interval
[.386184,.989868]. Both favor original FCSRL. DO difference-.002105 points
has interval[-.028684,.033553], crossing zero. Every quality component of the
complete adoption criterion fails. Do not interpret uncertain DO as equality
or use smoother commands as a replacement adoption criterion.

Native filtering uses52686.411455m3, about1.28 percent more than original,
with cost22.847625, NH423.598553 percent and DO.451579 percent. Native cost
and NH4 intervals are also unfavorable. The failure is not solely caused by
matching. Both native and adjusted outcomes and every air-only query remain
in the generated reports. Offsets are applied after filtering and subsequent
steps observe the actually executed clipped air, not the proposed dose.

The secondary mechanism metric behaves as intended: mean adjacent executed
command change falls from406.812460 to282.556246m3/h at matched air, with a
reduction in every seed. These exclude the initial jump and are command-rate
differences per15-minute change, not electrical energy. The initial jump is
reported separately using the public previous-air snapshot. Smaller command
variation does not imply better water quality under NN feedback and process
lag. Initial recorded air also affects the transient; this experiment does
not isolate steady behavior from reset initialization.

Gain1 preserves original action values/dtype and full native metrics; nonzero
offset, reset, terminal, serialization and previous-air conversion checks pass.
Gain.5 serialized native replay and independently rebuilt native/matched
actions, trajectories and command diagnostics pass for all seeds. Executed
volume reconstructed from commands agrees with environment accounting. No
implementation fault is identified by this coverage, which does not prove
that every possible fault is absent. Lag in a responsive controller is one
possible mechanism for the adverse result, not a proven sole causal explanation.
Do not tune a new gain or drop an unfavorable seed from these outcomes.

All intervals use five original policy seeds, SD ddof1 and10000 paired-seed
resamples. Model/world/noise draws and dates are not extra policy seeds.
Multiple reused-validation trials, the e40 DO onset/ranking limitation and
unverified real-plant counterfactuals remain. The fixed world, thresholds,
reward and full criterion are unchanged. This failure does not exhaust the
goal's alternatives or establish an external blocker.

artifacts.json verifies lee-e57-air-filter.zip,12,955,323 bytes, SHA256
364257cc50749880d3d899d103eb61230db3ebe92e41f1c2994bce9ac38e68ad,
in policy draft443333191, asset753539996. It preserves the five unchanged final
base checkpoints, filter configurations, required metadata, full logs, native/
matched/all-query outputs, independent audits and final report. Source data,
captured arrays and stochastic banks are excluded. Archive CRC/file hashes
and remote size/SHA256 verify. No e57 job remains queued. The next direction
revisits unresolved conditional DO event support in the environment instead
of retuning this failed filter on validation.
