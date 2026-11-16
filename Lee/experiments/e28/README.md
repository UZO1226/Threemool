# Lee/e28: Restore within-training seasonal coverage without touching test

## Hypothesis and fixed comparison

Holding out the latest30 training days deprives every earlier new environment
of the most recent operating weeks for weight fitting. The official protocol permits
training on all188 training days and keeps development40 and test41 distinct.
Test whether a season-spanning inner partition is more useful than the latest30
partition while keeping the fitting/calibration counts158/30 unchanged.

Preparation disproved an initially suspected minimum-temperature mechanism:
both the original and new fitting sets reach 14.0183 degrees C; the old inner
calibration minimum is14.3273. This experiment does not expand the minimum
temperature range. It instead tests coverage of later joint operating regimes
and less concentrated inner selection. Do not attribute any improvement to
a colder fitting minimum, and do not use the development minimum to fit it.

Choose 30 calibration indices by rounded linspace(0,187,30), fixed without
looking at concentrations, risks or development outcomes. The other158 dates
fit weights. Both sets remain inside official training. Inner selection is
interpolative across the training seasons, not a chronological forecasting
test. Future-within-training dates can influence earlier-date fits, so do not
call inner scores deployment generalization estimates. Official development
dates remain later than every fitting/calibration date.

Use the corrected e24 model, original .995 cap and original four-step innovations.
Do not combine e27's cap or e26's long blocks. Same loss, five seeds, 80-percent
date subsamples, 80 full-day epochs, common20/40/80 checkpoint selection and
16-scale inner search. Weight normalization and clipping fit the new158 dates.
For comparable calibration/selection and audit scores, freeze scoring SD to
e24's original158-date values. This scaling does not enter the new model's
training loss or feature normalization. The baseline and e16 adoption thresholds
therefore retain exactly their existing numeric meaning.

The old Data helper now indexes all official training dates, while deriving
fit statistics only from explicitly supplied fitting dates. It continues to
exclude all test timestamps. Past observed states remain causal context only.
Validate date counts/disjointness, cutoff and no development/test date in fit
or calibration before running. No official protocol file changes.

This is exploratory model development after repeated development inspection.
No claim of independent confirmation. Report separate NH4/DO and unchanged
gate. No new policies until an environment passes. A failed candidate does
not terminate the active goal. Source hashes/runtime/units/horizon/reward/cost
follow e24, with the inner partition difference explicitly recorded here.

```powershell
python Threemool/Lee/experiments/e28/prepare_reference.py --data kaggle_data/darrow_15min.pkl --reference results/e17_contractive/summary.json --out Threemool/Lee/experiments/e28/reference.json
python Threemool/Lee/experiments/e28/seasonal.py --data kaggle_data/darrow_15min.pkl --reference-summary Threemool/Lee/experiments/e28/reference.json --out results/e28_seasonal
```
