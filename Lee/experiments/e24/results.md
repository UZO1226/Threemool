# Lee/e24 corrected full-day results

No environment adoption and no new policy training. Correcting the dead NO3
initialization restores learning, but does not resolve NH4 safety calibration.
Execution commit 8997e8a. The independent initialization check reproduces the
old dead head and verifies equilibrium and nonzero gradients after repair.
The selected checkpoint is 40 epochs, with n1_d1 innovations, both selected
only on the last 30 training dates. All five members trained to 80 epochs.

| Forecast | Normalized recursive MAE | NH4 MAE (mg/L) | NH4 violation (%) | DO violation (%) |
| --- | --- | --- | --- | --- |
| Initial-state persistence | 0.9639 | 2.3361 | 30.00 | 0.00 |
| Original delta GBM | 1.0261 | 2.4823 | 28.48 | 1.80 |
| Corrected e24 | 0.8725 | 1.5135 | 5.87 | 1.27 |
| Observed transitions | | | 17.32 | 1.16 |

Compared with the original, normalized MAE improves 15.0 percent and NH4 MAE
39.0 percent. Compared with persistence, improvements are 9.5 and 35.2 percent.
These are forecast errors under recorded aeration, not treatment improvements.
NO3 MAE is 1.2488 rather than the historical dead-head value 2.5587. It remains
worse than simple persistence's 1.0560. Not every output improves.

NH4 frequency is underpredicted by 11.45 percentage points. Each of the five
members predicts only 4.57 to 6.56 percent. None meets both safety tolerances.
DO frequency error is only 0.109 points, and both air-order violations are zero
for the exercised common-noise probes. Sixteen draws give NH4 6.07 percent and
DO 1.26 percent, so extra stochastic draws do not remove the NH4 failure.

Event probability scores also matter. NH4/DO Brier scores are .12237/.01272,
against .21768/.02107 for the original and .32579/.01158 for persistence.
Thus DO event timing is not better than always predicting no DO violation,
despite much closer aggregate frequency. No blanket safety-accuracy claim is
made. All full metrics, including concentration CRPS, are in summary.json and
audit.json.

Ten thousand circular five-day block resamples of the 40 development dates
give NH4 frequency bias 95-percent interval [-14.49,-8.02] percentage points.
DO bias interval is [-.604,.751]. The paired normalized MAE change relative
to the original has interval [-.260,-.054], and NH4 MAE change [-1.309,-.630]
mg/L. These reused development dates support exploratory uncertainty, not a
fresh confirmation or policy-seed significance test.

| Observed-state reset interval | NH4 MAE | NH4 violation (%) | DO violation (%) |
| --- | --- | --- | --- |
| 1 step | .4008 | 15.14 | .51 |
| 4 steps | .8035 | 13.12 | 1.23 |
| 16 steps | 1.3460 | 8.79 | 1.27 |
| 48 steps | 1.4757 | 7.22 | 1.27 |
| 95 steps | 1.5135 | 5.87 | 1.27 |

Resets inject observed later states and are diagnostic forecasts, not valid
counterfactual rollouts or a policy evaluation. Cached NumPy and independent
Torch full-day forecasts agree within 2e-4. Exact reconstruction of the saved
selected score passes. The reset-95 check allows 1e-8 relative tolerance because
the measured initial state is float64 while the neural initializer uses float32.

Neither 80 epochs nor Powell's 120 evaluations proves convergence. The historical
e22/e23 scores must not be used to rule out neural models. e25 tests a simple
initial-state anchor on inner data before any new model adoption decision.
Official test observations remain unevaluated.

```powershell
python Threemool/Lee/experiments/e24/check_initialization.py --data kaggle_data/darrow_15min.pkl --out Threemool/Lee/experiments/e24/initialization_check.json
python Threemool/Lee/experiments/e23/full_day.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e24_fixed_initialization
python Threemool/Lee/experiments/e24/audit.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e24_fixed_initialization --old-diagnosis Threemool/Lee/experiments/e16/diagnosis.json --out Threemool/Lee/experiments/e24/audit.json
```

Use a new output directory for training reruns. Thread settings and CPU
packages follow e15/runtime.json. Source/data provenance follows e16 and
Lee/analysis/DARROW_SOURCE_REVIEW_20261112.md. Residual banks can be regenerated
from authorized training data and are excluded from the shared model archive.
