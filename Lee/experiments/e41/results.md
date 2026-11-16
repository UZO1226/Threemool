# Complete causal-history forecast result

Five model members finish80 epochs under source982124f. Inner calibration
selects epoch80 and n1_d1.5. No validation-based setting replacement is made.
Independent checkpoint reconstruction, full-day Torch/NumPy agreement and
common-noise air ordering pass. No final test or policy training is used.

The unchanged adoption gate fails NH4 calibration, DO calibration and member
agreement. No member passes both frequency tolerances. Normalized MAE.772093
and NH4 MAE1.388028 improve over original GBM1.026065/2.482256, but normalized
MAE is worse than unchanged e34.760109. Forecast NH4 frequency9.14145 percent
versus actual17.31579 has bias-8.17434 points, five-day block interval
[-10.85992,-5.40987]. DO2.55592 versus1.15789 has bias+1.39803 points,
interval[.60130,2.20663]. e40 remains the frozen historical policy environment.

NH4 AP.656955 versus e34.658540, AUC.858753 versus.868009 and Brier.103089
versus.103172 do not support a clear long-history benefit. The paired-date
Brier difference interval[-.004911,.003703] crosses zero. One-hour NH4 onset
matches7/89 versus e34's8/89. Exact matches2 versus0 are exploratory diagnostics,
not a basis to adopt a model failing the fixed criteria.

DO AP.012473 and AUC.514601 remain close to chance. All25 true DO onsets are
missed within one hour. DO Brier.016038 versus e34.015438 has difference
interval[-.001467,.002457]. Relative to persistence the DO Brier difference
is+.004459 with interval[.001137,.008547], worse on this reused validation.
e40/persistence event metrics reproduce the prior audit before comparisons.

Observed-state resets diagnose drift, not an available controller advantage:
resetting each step gives normalized MAE.384829 and NH4 frequency15.81184,
compared with daily free-running.772093 and9.14145. DO frequency shifts from
1.26184 to2.55592. Repeatedly revealing real observations changes the forecast
task; these reset numbers cannot be used as autonomous simulator performance.
They motivate separating one-step event learnability from recursive drift.

The parent-model equivalence, gradient and independent rollout checks passed.
The evidence points to a failed hypothesis under this fitting procedure rather
than a detected implementation error. It does not prove that every use of
longer operating history must fail. Added capacity still prevents attributing
any isolated metric difference solely to longer history.

summary.json, audit.json and event_reference_audit.json preserve all settings,
member frequencies, date intervals and metrics. artifacts.json identifies
the draft-release checkpoint/log archive with remotely verified size and SHA256.
Source data and the residual bank are excluded. Re-running the recorded
training command recreates the latter. The next diagnostic will test next-step
NH4/DO event learnability from current observations before further long-rollout
changes. No adoption gate is lowered, and no environment is replaced here.
