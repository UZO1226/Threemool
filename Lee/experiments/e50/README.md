# Lee/e50: Quantile training exposed to accumulated state error

Execution snapshot: sourcee3a0e9d has completed all105 model fits in
results/e50_quantile_exposure. The same live shell is evaluating the fixed
inner20 settings and then runs independent reconstruction and event audits
on success. Console log is results/e50_quantile_exposure_run.log. This is not
a live monitor or an adoption result; inspect the actual process/tool handle
before waiting or restarting. Never duplicate because a log is quiet.

## Fixed hypothesis

e49 demonstrates useful observed-state NH4 prediction but much poorer daily
free-running e40 forecasts. Test whether fitting e40's quantile transition
with exposure to its own accumulated state errors reduces that gap. Retain
observed-state examples with equal total weight. This is a new environment
candidate, not an agent extension and not a replacement of historical e40.

Freeze e40 weights before generating training examples. On each of the158
fitting dates, each member rolls95 steps from the real initial state with
recorded air/exogenous inputs, alpha.5 and deterministic median quantiles.
No PIT bank, calibration labels, validation labels or final test outcomes are
used to construct exposure states. For every fitting transition, pair the
member's pre-transition forecast state with that date's recorded exogenous
inputs and air. The supervised target remains the real next concentration.
This is model-error input augmentation under recorded actions, not new physical
or counterfactual plant observations. It may fail when a distorted input state
has a different true successor. Preserve that limitation in interpretation.

For each member's original80-percent date subset, fit safety outputs on a
50/50 mixture of original observed-state and exposed-state examples. There is
one fixed augmentation/refit round, no searched mixing coefficient or repeated
validation-dependent round. Keep NO3 fitting unchanged on observed-state rows.
Five model seeds, five quantiles,105 fits,200 iterations,15 leaves, leaf40,
L2=1,learning rate.05 and all action/state monotonic signs remain e40's.
Doubling safety examples also changes effective minimum-leaf support; a
positive result would require an observed-row duplication control before
attributing improvements specifically to exposure. Retain the same e40 scoped
quantile leaf-bound repair, checking function restoration.

Before full fitting, verify that zero augmentation returns exactly the
original X/targets, all first-step exposure states equal real initial states,
future concentration scrambling changes targets but not exposure inputs,
all arrays are finite and air is inside the original cap. Record differences
between exposed and observed states. The parent freeze verifies all weight
and inference hashes. Do not silently refit or change that parent.

After refitting, reuse the original e38/e40 inner30 search unchanged:
20 noise/alpha settings, PIT estimation on inner30, original proper score,
eight draws,16-draw sensitivity, independent sklearn full-day reconstruction,
coordinate and recursive common-noise air ordering and unchanged e16 gate.
Report NH4/DO error, frequency, Brier, AP/AUC, onset0/1/4 and date uncertainty
against e40, original and persistence. Gate failure remains failure. No
threshold tuning, model adoption or controller training is automatic.

This is exploratory reused validation. Five model members are not five policy
seeds. Runtime, original FCSRL source/license, units, generated-influent caveat,
reward, cost,95-step horizon and final test preservation follow e40/e42.
Keep raw inputs, exposure arrays and banks outside Git. Upload only model/log
artifacts to the existing environment draft and record verified checksums.

```powershell
python Threemool/Lee/experiments/e50/exposure.py --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --preflight-out Threemool/Lee/experiments/e50/preflight.json
python Threemool/Lee/experiments/e50/exposure.py --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e50_quantile_exposure
python Threemool/Lee/experiments/e40/audit_quantiles.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e50_quantile_exposure --out Threemool/Lee/experiments/e50/audit.json
python Threemool/Lee/analysis/audit_quantile_events.py --data kaggle_data/darrow_15min.pkl --runs results/e50_quantile_exposure --freeze Threemool/Lee/experiments/e42/environment_freeze.json --audit Threemool/Lee/experiments/e50/audit.json --out Threemool/Lee/experiments/e50/event_reference_audit.json
```

preflight.json passes before fitting. Median exposure changes NH4 inputs by
mean absolute1.36929mg/L on fitting dates, with maximum10.74346mg/L. This is
substantial model error, not small measurement noise; retain the possibility
that the paired true targets become inappropriate supervision for those states.
