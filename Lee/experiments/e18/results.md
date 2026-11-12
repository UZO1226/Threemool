# Lee/e18 results

Inner proper scoring selects NH4 innovation scale 1 and DO scale 1. The
probabilistic forecast still fails the unchanged e16 adoption gate. No new
policy training is run on it. No test observations are evaluated.

| Development metric | Deterministic reference | Selected innovation model |
| --- | --- | --- |
| Normalized trajectory MAE | 0.6427 | 0.7351 |
| NH4 trajectory MAE | 1.7459 | 1.7509 |
| NH4 violation prediction (%) | 4.16 | 5.64 |
| DO violation prediction (%) | 0.00 | 1.77 |
| Normalized CRPS | 0.5931 | 0.5162 |
| NH4 Brier score | 0.15447 | 0.14858 |
| DO Brier score | 0.01158 | 0.01261 |

Actual NH4 and DO frequencies are 17.32 and 1.16 percent. Distributional
scoring and NH4 Brier improve, but NH4 frequency error remains 11.67 points.
DO frequency calibration improves, while DO Brier worsens: matching an
aggregate event rate does not ensure prediction of when events occur.
Zero structural members meet both calibration tolerances.

With 16 rather than eight noise draws per member, NH4 frequency is 5.65
percent and DO 1.83 percent. The NH4 failure is stable to that Monte Carlo
change. DO error changes from 0.61 to 0.68 points, crossing the old-model
non-worsening threshold 0.642. There is no robust environment adoption.

The empirical CRPS identity, zero-noise equivalence, common-innovation action
ordering and initial-state contraction checks pass. Innovation paths and
model members are not policy training seeds. Residuals come only from the
30 training calibration dates and are not included in Git.

The NH4 scale selected by inner score is the grid's upper boundary. e19
prospectively extends that inner scale grid and tests a fit-only temperature
trend. The failed e18 choice is preserved and is not replaced by a candidate
chosen from development outcomes.

## Reproduction

Execution commit 90a6eef. Runtime follows e15. Full residual banks and summary
are at results/e18_innovations; their hashes are in the versioned summary.
The residual bank is derived from source observations and is not redistributed.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python Threemool/Lee/experiments/e18/innovations.py --data kaggle_data/darrow_15min.pkl --contractive-summary results/e17_contractive/summary.json --old kaggle_data/darrow_surrogate_gbm.pkl --out results/e18_innovations
```

Use a new output directory for a rerun. This does not change official v1.1
or establish counterfactual response validity for the actual treatment plant.
