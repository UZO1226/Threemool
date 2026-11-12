# Lee/e20 results

The inner procedure selects contraction bound .95 and innovation scales
NH4 1, DO 1.5. It improves DO frequency but fails the unchanged NH4 adoption
gate. No new policy is trained on this environment, and test remains unused.

| Development metric | Original delta GBM | Selected coupled model |
| --- | --- | --- |
| Normalized trajectory MAE | 1.0261 | 0.8134 |
| NH4 trajectory MAE | 2.4823 | 1.7568 |
| NH4 predicted frequency (%) | 28.48 | 5.51 |
| DO predicted frequency (%) | 1.80 | 1.32 |

Observed frequencies are NH4 17.32 and DO 1.16 percent. Selected NH4 error
is 11.81 percentage points, and no structural member passes both calibration
tolerances. DO error falls to 0.16 points. The failure is not resolved by
combining sign-consistent coupling with contraction. This is not proof that
all coupled models are inadequate.

Both matrix families pass fast/sklearn and weighted initial-state contraction
checks. Their five member matrices, convex-solver outcomes, bias calibration
and inner distribution scores are preserved in summary.json. Noise paths are
not additional training seeds. No development outcome selected the matrix
or noise scales. Multi-step latent dynamics is tested next in e22.

## Reproduction

Execution commit 05a9df0, captured before fitting. CPU and packages follow
e15/runtime.json. The same first 158 training dates fit models and last 30
calibrate them. Artifacts are local under results/e20_coupled; versioned
summary retains model paths and hashes. Complete calibration traces are in
the runtime summary, with trace bodies omitted from the compact Git copy.

```powershell
python Threemool/Lee/experiments/e20/coupled.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e20_coupled
```

Use a new output directory for rerun. No source observations are committed.
