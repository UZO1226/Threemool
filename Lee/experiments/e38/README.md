# Lee/e38: Conditional quantile transitions with a calibrated residual copula

Historical result correction: the installed quantile leaf-update behavior does
not preserve all nominal feature constraints. e38 fails recursive air ordering
despite exact fast/sklearn agreement. The sorting argument below is conditional
on component monotonicity and does not repair that missing premise. See e40's
independent reproduction and bounded-update correction. No e38 model is adopted.

## Hypothesis and factual diagnostic

e34 misses NH4 events both when initial NH4 is safe and when it is unsafe.
On development28 safe-start days, actual/predicted frequencies are14.21/7.61
percent; on12 unsafe-start days they are24.56/12.59. Merely retaining unsafe
initial values cannot repair the safe-start misses. Point-forecast models with
state-independent additive noise may also miss state-dependent transition tails.
Fit conditional transition quantiles directly rather than another point model.

Use HistGradientBoostingRegressor loss=quantile at .05,.25,.5,.75,.95 for NH4
and each DO output. NO3 uses one median model, reused across quantile slots;
NO3 does not feed back into the safety outputs. Five members each fit a seeded
80-percent date subset of the original158 fitting dates. Use200 iterations,
rate .05,15 leaves, minimum40 samples/leaf, L2=1 and no early stopping. Date
subsets differ from e16's row subsets, so this is a family comparison, not
a pure loss-only causal ablation. Fit105 unique models, with at most4 independent
CPU fitting tasks and one OpenMP thread each.

Preserve e16's cooperative state/air signs. Every quantile has the same signs.
Sort quantile predictions pointwise to remove crossings, clip to the fitting
physical range and interpolate at fixed sampled probability levels. Sorting
and interpolation preserve pointwise ordering when all component functions
have the same monotonic directions. Thus common-noise recursive air order
remains testable. This prior does not identify actual treatment-response magnitude.

Estimate probability-integral-transform values on the inner30 teacher-forced
transitions. Randomize exact quantile ties inside their CDF interval, including
endpoint atoms from clipped tails. Store this row-level bank outside Git.
Sample within-day blocks1,4,16,95 to preserve estimated cross-output/time
dependence. Compare an independent-uniform control as well. NO3 stays median.
The quantile spread itself depends on current state and exogenous inputs.
Support beyond the two extreme quantiles remains unmodeled and is reported.

Select noise mode/block and persistence blend alpha0,.25,.5,.75 by the original
inner proper score. Next state is alpha times current state plus(1-alpha) times
the sampled next concentration. This preserves the response signs. Inner20
configurations use8 draws/member. Audit only the selected candidate and the
uniform alpha0 control, with16-draw sensitivity, unchanged e16 gate and common-
noise .75/1.25 air probes. Do not tune on development outcomes.

Before scoring, independently check interpolation against numpy.interp, exact-
atom CDF intervals, numeric fast/sklearn identity on observed and perturbed
features, absence of NO3 feedback splits and deterministic noise generation.
Report one-step coverage as a diagnostic distinct from recursive evaluation,
plus separate NH4/DO event probability/timing and uncertainty. The gate is not
relaxed. No agents before a validated environment, no final test evaluation.

Official split188/40/41, model-fit158/inner30, prepared/raw hashes, original
FCSRL commit,95-step horizon, reward/cost/action limits, physical units and CPU
package versions follow e24. These are model seeds, not policy-training seeds.
Development remains exploratory reused audit. The sklearn1.9.1 primary API
documents pinball loss and feature monotonic constraints:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

```powershell
python Threemool/Lee/experiments/e38/quantile_dynamics.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e38_quantile
```
