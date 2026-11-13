# Lee/e49: Separate next-step event learnability from recursive drift

## Fixed diagnostic question

e41's one-step observed-state reset substantially reduces frequency bias, but
daily free running still fails. e40 also has weak DO event ranking. Test whether
current measured state plus recorded exogenous variables and air can predict
next-step events before changing another simulator. This is a supervised
diagnostic, not an adopted transition model or an FCSRL extension.

Use the existing158 fit dates only. Keep30 inner-calibration and40 validation
dates separate for reporting; no hyperparameter, threshold or model selection
uses either. Final test41 is excluded before array construction. Each date has
95 within-day transitions. Targets are next NH4>4 and maxDO>2. Features are
the original24 current-state/exogenous/current-air/previous-air inputs.
The first previous air repeats current air, matching historical environment
semantics. Also fit an otherwise identical no-state19-input diagnostic to
measure reliance on current concentrations. These are observational features;
generated influent availability and air endogeneity retain earlier caveats.

Fit five date-subsampled HGB classifiers per label and feature set: model seeds
0..4,80 percent of fitting dates without replacement,200 iterations, learning
rate.05,15 leaves, minimum40 observations per leaf,L2=1, no early stopping,
no class reweighting, no monotonic constraints and no probability calibration.
These fixed settings mirror earlier HGB capacity; they are not searched.
Average member probabilities. Save all models and date subsets. Model members
are not policy seeds and no controller is trained.

Compare current-state persistence, frozen e40 observed-state next-step forecasts,
and e40 daily free-running forecasts on identical targets/noise. Reproduce
prior e40 validation Brier/AP/AUC for the free-running reconstruction. Respect
the action cap and previous-zone semantics. Observed-state resets are diagnostic
and cannot be counted as free-running simulator performance.

Report each label separately: Brier, AP, AUC, threshold.5 precision/recall,
probability mean, threshold event frequency, reliability bins and within-day
onset matching at tolerances0/1/4. Also report next-step metrics restricted to
currently nonviolating observations, where the target is a genuinely new onset.
Otherwise persistence of existing violations could masquerade as onset skill.
Use paired10000 circular five-day Brier intervals on validation dates. All
validation results remain exploratory because this period has been reused.
No adoption gate is lowered or evaluated as passed by this diagnostic.

## Reproduction

```powershell
python Threemool/Lee/experiments/e49/diagnose.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out results/e49_next_step
```

Raw data and probability arrays stay outside Git. Store compact reports under
Lee/e49 and upload model/log artifacts to the existing environment draft with
checksums. Original FCSRL commit/license and measurement interpretation follow
e42 and the source review. Only Lee/ changes on lee/darrow-safety-calibration.
