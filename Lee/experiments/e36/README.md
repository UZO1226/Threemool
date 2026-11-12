# Lee/e36: Align shared bias calibration with the proper selection score

## Hypothesis and frozen parent

Earlier bias calibration minimizes normalized mean absolute error plus soft
aggregate-frequency errors, while checkpoint/noise selection uses normalized
CRPS plus separate NH4/DO Brier scores. That objective mismatch remains an
executable alternative. e34 has the lowest inner proper score among the current
neural candidates on the same inner30 dates, so freeze that parent and test
proper-score calibration without changing weights or noise.

Keep NO3 bias fixed and optimize four shared safety-output biases within the
original bounds: NH4 +/-.5 and each DO +/-.25 mg/L per update. Use the full
coupled stochastic trajectory and the original normalized CRPS plus both Brier
scores. The objective is not separable across outputs because state feedback
couples them. Do not pretend that separate scalar profile optima solve it.

Compare the parent and zero safety-bias controls on inner30 first, then run two
coordinate sweeps with21 grid points per safety output over the full original
bounds. Refine within one coarse spacing using9 points per output for two
further sweeps, with the last spacing divided by4. Keep the current point in
every grid and memoize objective calls. Retain the best inner score. This is a
finite deterministic coordinate search, not a proof of the global optimum.
It includes the unchanged parent, so inner optimization alone is not evidence
of generalization. No bounds or validation thresholds are relaxed.

Use common noise for every candidate, source e34's selected n1_d1 scales and
four-step residual bank. Bank re-estimation is excluded to isolate shared
calibration. First158 days fit the frozen weights; inner30 calibrate/select;
development40 is exploratory reused audit; official test41 stays unevaluated.
No new model fits or policy seeds are counted. Preserve model-kind context,
.995 state-matrix cap, action signs,95-step horizon, reward/cost/action limits,
units, source hashes, original FCSRL commit and runtime from e24/e34.

Assert exact unchanged-parent score reconstruction on inner and development
data. Report separate probability/event metrics, unchanged adoption gate,
16-draw sensitivity and independent full-day audit. The audit loads checkpoints
and noise from --parent-runs while using the new summary's bias, with SHA256
checks. No policy training until an environment passes.

```powershell
python Threemool/Lee/experiments/e36/proper_calibration.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent results/e34_joint_teacher --out results/e36_proper_calibration
python Threemool/Lee/experiments/e24/audit.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e36_proper_calibration --parent-runs results/e34_joint_teacher --old-diagnosis Threemool/Lee/experiments/e16/diagnosis.json --out Threemool/Lee/experiments/e36/audit.json --model-kind context
```
