# e59: Observed-state versus recursive DO prediction diagnosis

This diagnostic follows the completed e58 failure. It is not a fitted model,
new controller, candidate selection or adoption attempt. Preserve all earlier
failures, e42 freeze, original outcomes and reserved41 test days.

Before running, fix these comparisons: training158, calibration30 and
validation40 marginal distributions for current NH4, maximum current DO,
temperature, flow and current total air; observed next-step NH4/DO prevalence;
validation fractions outside fit/calibration feature ranges. Do not inspect
test rows, fit new thresholds or select context bins after outcomes.

For e51 and e58 retain their already selectedpit_b95_a0.25, original calibration
banks, five members, eight draws, all30 calibration/all40 validation days and
the same day/member/draw keys. Reset the current state to the true observation
for each one-step forecast. Compare resulting NH4/DO probability Brier, AP/AUC,
frequency and new-onset0/1/4-step matching with the stored full-day validation
results. No future observation may enter the prediction input. Calibration
uses the same bank-estimation dates and is descriptive, not held-out evidence.

Authenticate original weights, summaries, dataset and unchanged split. Require
observed input/target alignment with the existing day arrays and exact
non-DO quantile backbone equality. Predictions may use model outputs only;
actual next-state labels are used for scoring afterward. Do not infer a plant
counterfactual from an observed-state diagnostic, or claim that resetting
states is a deployable controller/world improvement.

Report all fixed comparisons. These reused validation diagnostics can guide
a later prospectively specified hypothesis, but must not rescue e58, tune its
tail, lower the adoption criterion or justify training an agent on it. No
post-outcome context subset is used as a headline success. Source observations,
probability arrays and stochastic banks stay local and out of Git/artifacts.
