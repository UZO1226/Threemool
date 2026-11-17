# e76: Persistence plus learned DO change, with unchanged admission conditions

Prospective model hypothesis after complete e75 rejection,2026-11-17.
Absolute next-DO regression fails persistence comparisons or event misses in
four of five seeds despite strong constant-baseline gains. Test retaining
the observed current maxDO as a fixed baseline and learning only next DO
minus current maxDO. This is a model-development hypothesis, not a proven
cause of the earlier failure. Preserve every e75 result and all admission
conditions; do not recalibrate after seeing held performance.

Use the same authenticated captures, current obs11 plus logged action,
original401280-step NNs, data source and update partition:1 through40 fit,
41 through52 calibrate,53 through66 held. Input dimension remains12 and no
future observation enters a feature. Target subtraction uses current maxDO
already in the input. Negative predicted changes must remain possible.
The physical point forecast is max(0, current maxDO plus predicted change).
Do not clip changes to nonnegative values or forecasts to[0,1].

Keep the same squared-error HistGradientBoostingRegressor,200 iterations,
learning rate.05,15 leaves, min_samples_leaf40, L2=1, no early stopping and
random_state7500. Keep negative NH4 and positive previous/proposed-air
directions. Current DO coordinates become unconstrained in the change model:
subtracting max(current DO) makes their old positive absolute-level direction
inappropriate, especially when the maximum sensor changes. The complete
point forecast remains monotone in action, which must independently verify.
This coherent persistence-residual change is not an isolated ablation of
each resulting constraint implication.

Keep e75's empirical95th-percentile calibration margin, higher order statistic
and nonnegative margin floor, plus the same calibrated persistence and
fitting-mean controls. Keep every all-five admission gate unchanged: exact
provenance/rows, original/compiled raw agreement1e-12, monotone action probes,
paired-date MSE difference interval upper endpoints below zero versus both
controls, held upper coverage at least.95, and next-DO false negatives no
greater than calibrated persistence. Use10000 date resamples with RNG7502.
Report all physical errors, coverage uncertainty, next events, onsets and
ranking diagnostics. Global coverage is not rare-event recall or calibrated
event probability. Source captures remain local.

The previously inspected held update block is now exposed model-development
data, not new independent confirmation. Reused validation40 also stays
exploratory and test41 remains unused. No validation label or action quality
fits this model. The RL interaction budget is unchanged; report the additional
offline supervised fit, model parameters and CPU time separately.

Before any controller evaluation independently audit each raw change and
physical forecast, calibrated margin and every admission decision. If any
seed fails, retain all five models and reject the conditional controller stage.
Do not select seed3, mix e75/e76 per seed, pool model/date samples as policy
seeds or loosen gates. If all models are admitted, use e75's already declared
shielded matched-anchor controller and complete controls, actual-air comparison
and independent trajectory audits without changing settings. If its original
block passes, require the separate exposed5 through9 block and then actual
new training seeds10 through14 before adoption. No new environment is adopted.
