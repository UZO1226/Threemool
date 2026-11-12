# Lee/e30: Tail-weighted environment fitting, unchanged evaluation

## Hypothesis

e29's extra240 epochs reduce fitting loss but worsen inner checkpoint scores.
The unchanged warm-start control wins. More fitting of the same average loss
is therefore not supported by that trial. Test whether rare safety-event
trajectories receive insufficient regression weight.

Change only the environment training loss. For NH4 target>4, multiply its
concentration regression weight by4. For any target DO>2, multiply each of
the three DO regression weights by4. NO3 weight stays1. Normalize weights
within each output over each batch so outputs retain equal aggregate scale.
Use .5 weighted normalized absolute error plus .5 weighted normalized squared
error, and retain the original .5 sum of NH4/DO soft-Brier terms unchanged.
This is deliberately tail-emphasized point fitting, not a proper distribution
score or a changed definition of safety. Record the risk of overprediction.

The factor4 is one prespecified moderate weight, not chosen from development
outcomes. Keep the corrected e24 model, .995 cap, first158/last30 inner split,
five paired fitting seeds and80-percent date subsets, batch16, Adam .001,
80 full95-step epochs, common20/40/80 inner checkpoint selection, original
four-step innovations and16-scale inner search. Do not combine e27, e28 or
e26 changes. Proper inner selection and e16 adoption thresholds stay unchanged.
Only training targets determine the event weights. Development40 is reused
exploratory audit; official test41 is not evaluated.

The weighted-regression check uses equal concentration errors on two rows,
one rare and one normal. The rare-row gradient must be exactly4 times larger.
When no event occurs, the regression term must equal the unweighted baseline.
Run this before fitting. Keep the NO3 initialization regression in force.

Units, horizon, reward, cost, action bounds, source/data hashes, FCSRL commit
and CPU package versions follow e24. No new policy training until an environment
passes. Failed weighting is retained and does not terminate the active goal.

```powershell
python Threemool/Lee/experiments/e30/check_loss.py --out Threemool/Lee/experiments/e30/loss_check.json
python Threemool/Lee/experiments/e30/tail_weighted.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e30_tail_weighted
```
