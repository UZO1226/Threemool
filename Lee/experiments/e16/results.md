# Lee/e16 results

The cooperative family fixes the recursive sign failure but does not pass
the preregistered environment adoption gate. No new agent training is run
on this model. Final team test observations remain unused.

| Model | Normalized recursive MAE | NH4 MAE | Predicted NH4 violation (%) | Predicted DO violation (%) |
| --- | --- | --- | --- | --- |
| Old delta GBM | 1.0261 | 2.4823 | 28.48 | 1.80 |
| e14 direct 7 | 0.9049 | 1.9448 | 26.98 | 0.17 |
| Cooperative 7 | 0.9838 | 4.1881 | 65.97 | 0.76 |
| Calibrated cooperative 7 | 0.6820 | 2.3437 | 8.13 | 0.17 |
| Cooperative 15 | 1.0437 | 4.7158 | 70.82 | 0.48 |
| Calibrated cooperative 15 | 0.7036 | 2.3845 | 8.51 | 0.24 |

Actual transition-level violation frequencies are NH4 17.32 percent and DO
1.16 percent. The full summary is authoritative for every value, member and
horizon. Normalization uses the first 158 fitting days' standard deviations,
so its numeric scale differs from e14. Raw concentration MAEs are comparable.

The candidate selected on inner calibration alone is calibrated 15. It
improves normalized MAE by 31.4 percent but NH4 MAE by only 3.9 percent.
NH4 frequency error 8.81 percentage points exceeds the 5-point tolerance;
DO error 0.92 points worsens the old model's 0.64-point error. Zero members
satisfy both individual tolerances. Calibration optimization reached its
120-evaluation limit in both candidates; convergence is not established.

The raw and calibrated cooperative models have zero recursive air-order
violations across all validation dates and members in the exercised action
range. The old model's maximum NH4/DO order violations are 0.267/2.192 mg/L;
e14 direct 7 has 5.475/1.640 mg/L. Thus monotonic one-step air terms did not
guarantee trajectory ordering in the earlier models, and the structural fix
does not by itself ensure accurate concentrations or safety frequencies.

Five-day moving-block resampling across development dates supports reduced
normalized prediction error for the selected candidate: mean difference
-0.3225, 95 percent interval [-0.4084, -0.2345]. This is not a safety calibration
test or a policy-training-seed confidence interval. A lower predicted violation
frequency is not an improvement when the actual frequency is underpredicted.

## Data and seasonal diagnostic

Both raw input SHA256 values match protocol v1.1. Prepared training and
development columns match the raw measurements exactly to checked tolerance.
12.5 percent of development timestamps are below the training temperature
range. Thirteen development days include at least one such timestamp.
On those days the selected model predicts NH4 violation 17.89 percent versus
25.99 percent actual, and DO 0 versus 2.11 percent actual. On the other
27 days it predicts NH4 3.98 versus 13.14 percent actual. Temperature range
shift is therefore a real limitation, but cannot alone explain the failures.
See data_audit.json and diagnosis.json.

## Next hypothesis

The nonlinear feedback still allows small biases to alter the long-run state
substantially. e17 tests fixed contractive state feedback with nonlinear
exogenous forcing. This is a new prespecified model family, not a relaxed
adoption threshold or a new policy tweak. Fit and calibration remain confined
to training dates. Lack of demonstrated counterfactual validity remains.

## Reproduction

Execution commit 0845900. Original FCSRL source remains unchanged. Commands
use the existing project Python environment (e15/runtime.json), CPU, and
OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1.

```powershell
python Threemool/Lee/experiments/e16/data_audit.py --data kaggle_data/darrow_15min.pkl --raw kaggle_data/darrow --out Threemool/Lee/experiments/e16/data_audit.json
python Threemool/Lee/experiments/e16/cooperative.py --data kaggle_data/darrow_15min.pkl --old kaggle_data/darrow_surrogate_gbm.pkl --direct results/e14_audit/direct_7.pkl --out results/e16_cooperative
python Threemool/Lee/experiments/e16/diagnose.py --data kaggle_data/darrow_15min.pkl --runs results/e16_cooperative --out Threemool/Lee/experiments/e16/diagnosis.json
```

Use a new output directory on rerun. Local models and complete optimization
traces are at results/e16_cooperative; exact hashes and trace locations are
in summary.json. No raw data is committed. Summary calibration traces are
trimmed only for size, with the complete trace checksum retained.
