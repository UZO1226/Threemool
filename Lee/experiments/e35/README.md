# Lee/e35: Pre-episode innovation assimilation

## Prospectively specified hypothesis

Replay errors in the already observed four hours before an episode may reveal
unmodeled current conditions more directly than a new neural encoder. Keep
e24's corrected40-epoch models, global bias and selected innovations frozen.
For each member/day, initialize latent states before the past16 transitions,
then use recorded past air/exogenous inputs and teacher-forced observed states
to estimate their mean pre-clipping residual. No concentration after episode
start enters this estimator. Context preceding that window initializes hidden
states; its time-of-day support is a limitation to check, not future supervision.

Clip mean residuals to +/-.5 mg/L for NH4 and +/-.25 for each DO output, with
NO3 correction zero. These are the original bias-calibration bounds, fixed
before scoring. Add a fixed member/day correction to future forcing, independent
of future actions. Select NH4 gain0,.25,.5,1,2 and common DO gain0,.25,.5,1 by
the existing inner proper score on last30 training dates. Keep the parent
four-step residual bank and selected n1_d1 scales fixed to isolate assimilation.
Zero gain must exactly reproduce the parent on inner and development data.

Future targets may be used for scoring only. Check that scrambling all future
concentrations of a probe episode leaves its estimated correction exactly
unchanged. Assert all estimator targets occur at or before the episode start.
Report history depth and residual magnitudes, the unchanged e16 gate, common-
noise air ordering,16-draw sensitivity, separate event Brier/timing/reliability,
and date-block uncertainty. Bounds/gains are not tuned to development outcomes.

Same official188/40/41 split, first158 weight fitting, inner30 selection,
mg/L and m3/h units,95-step horizon, reward, summed cost and action limits as
e24. Test remains unevaluated. Five frozen model members are not policy seeds.
No new policy training before an environment passes. Development is reused
exploratory audit. This is data assimilation in a surrogate, not a calibrated
Tilburg process model or proof of counterfactual treatment response.

```powershell
python Threemool/Lee/experiments/e35/assimilate.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent results/e24_fixed_initialization --out results/e35_assimilation
```

CPU packages, raw hashes, preparation and original FCSRL commit follow e24.
Use OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1. Full corrections
and residual diagnostics remain outside Git if they contain row-level values;
the compact summary contains only aggregate checks and metrics.
