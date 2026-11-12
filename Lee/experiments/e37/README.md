# Lee/e37: Center pre-episode innovations before adapting the frozen model

## Hypothesis from e35's failure

e35 selects zero NH4 gain. Its mean pre-episode NH4 residual is .0895 mg/L on
inner dates but .2687 on development. The frozen global bias was optimized for
recursive forecasts, not zero-mean teacher-forced transition residuals. Applying
raw innovations can therefore count a persistent calibration offset twice.
Test deviations from the model's training-calibration residual mean instead.

Keep the e35 parent e24 and every other setting fixed. Estimate each member's
mean pre-episode innovation across inner30 dates, subtract this fixed training-
only mean from each episode's raw mean innovation, then apply the original
physical bounds and original20 NH4/DO gain pairs. Development corrections use
the same frozen center. No development labels estimate it. NO3 remains zero.
The inner mean is offline calibration across all30 dates, not independent
chronological confirmation for those same dates.

Use the same proper score, unchanged adoption gate, zero-gain reconstruction,
future-target scrambling negative control, common-noise ordering,16-draw
sensitivity and separate event/uncertainty reporting. Centering is the only
new factor; no decay, new parent, new weights or re-estimated noise bank.
This hypothesis is not a claim that the e35 implementation was buggy.

Official test remains unevaluated. Units, source/data hashes, runtime, original
FCSRL commit, reward/cost/action bounds and95-step horizon follow e35/e24.
Five frozen model members are not new model fits or policy-training seeds.
No new agents until an environment passes. Failure continues the active goal.

```powershell
python Threemool/Lee/experiments/e37/centered_assimilation.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent results/e24_fixed_initialization --out results/e37_centered_assimilation
```
