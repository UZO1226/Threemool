# Lee/e51: Calibrate PIT against the actual blended transition

## Prospectively fixed hypothesis

e50's forecast-state augmentation improves MAE but worsens NH4 Brier and fails
adoption. Return to frozen e40 weights, which retain observed-state skill.
Its PIT bank is estimated from unblended predicted quantiles, but a transition
with nonzero alpha is alpha times current state plus1-alpha times a sampled
quantile. For alpha.5 these are different conditional distributions. Test PIT
calibration to the actual blended one-step distribution. The earlier shared
bank is an intentional modeling choice, not an established implementation bug.

Use no new weight fitting. Verify e42's parent freeze and copy exactly the five
e40 quantile checkpoint files to a new results folder for independent auditing.
Keep first158 fitting dates, last30 inner-calibration dates, validation40 and
unused final test41. Quantile repair, action/state signs and all parent weights
are unchanged. Original FCSRL source/license, units and generated-influent
limitations follow e40/e42. No policy training is part of this experiment.

For each original alpha0,.25,.5,.75, evaluate the frozen conditional quantiles
on observed inner-calibration inputs. Transform every quantile to
clip(alpha*current_state+(1-alpha)*quantile,0,original_hi), then estimate PIT of actual next state
under those transformed quantiles using the original tie RNG seed3801.
Preserve NO3 PIT.5 and all original endpoint/interpolation conventions.
Estimate a separate bank for each alpha. No state-dependent neighbor selection,
additional bias, changed support or new hyperparameter is introduced.

Reuse the original20-setting proper-score selection: uniform block1 plus
PIT blocks1/4/16/95, crossed with the four alphas. Each PIT setting uses its
matching alpha bank. Uniform settings remain unchanged. Keep original day/member/
draw noise keys, eight primary draws and16-draw sensitivity. No validation
outcome selects alpha or block. The same calibration dates estimate the bank
and choose settings as before; this is resubstitution-based exploratory
calibration, not an independent confirmation of coverage.

Before adoption, require alpha0 bank equality with frozen e40 and reconstruction
of all eight unchanged inner controls: four uniform-alpha settings and four
PIT-alpha0 block settings. Verify copied weight hashes, finite PIT values and
exact interpolation consistency for affine transformed quantiles. Record
central-interval coverage and endpoint PIT mass for every alpha. Blending
narrows support and PIT recalibration cannot recover outcomes outside it.

After selection save only the selected bank as pit_bank.npz, so the existing
independent e40/audit_quantiles.py and generic audit_quantile_events.py can
reconstruct the candidate. Additional banks remain local artifacts excluded
from Git. Keep the unchanged e16 gate, coordinate/recursive common-noise air
order, separate NH4/DO frequency/error/Brier/AP/AUC/onsets0/1/4, persistence and
original/e40 references,10000 five-day paired intervals and reused-date caveats.
Passing point-frequency criteria alone does not establish DO onset accuracy.
No automatic model replacement or new policy fitting follows a pass.

Raw observations and PIT arrays stay outside Git. Keep compact reports and
scripts under Lee/e51 and model/log artifacts in the existing environment
draft, with verified checksums and reconstruction commands. The active goal
continues if this alternative fails. Historical e40 policy comparisons remain
frozen, and no final test date is used for development.

Status: implementation and kernel preflights complete before outcome evaluation.
preflight_initial_failure.json preserves the initial assumption failure,
preflight.json checks the corrected kernel, and preflight_support.json adds
the actual out-of-bound NO3 inputs and CDF endpoints to interpolation probes.
All corrected preflights pass. No transition weights are fitted.

Execution snapshot: source1a2634d runs the fixed inner20 comparison in
results/e51_blended_pit, with log results/e51_blended_pit_run.log. The same
shell runs both independent audits on success. Initial uniform-alpha0/.25
controls reconstruct the parent's metrics within the required numeric
tolerance. Remaining controls and outcomes are pending at this snapshot.
Inspect the actual process/tool handle before waiting or restarting; do not
duplicate a quiet run. No adoption or controller improvement is claimed.

```powershell
python Threemool/Lee/experiments/e51/calibrate_blend.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --preflight-out Threemool/Lee/experiments/e51/preflight_support.json
python Threemool/Lee/experiments/e51/calibrate_blend.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --out results/e51_blended_pit
python Threemool/Lee/experiments/e40/audit_quantiles.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e51_blended_pit --out Threemool/Lee/experiments/e51/audit.json
python Threemool/Lee/analysis/audit_quantile_events.py --data kaggle_data/darrow_15min.pkl --runs results/e51_blended_pit --freeze Threemool/Lee/experiments/e42/environment_freeze.json --audit Threemool/Lee/experiments/e51/audit.json --out Threemool/Lee/experiments/e51/event_reference_audit.json
```

Use fresh output paths for repeating a recorded command. Correcting PIT does
not expand the distribution's bounded quantile support. At alpha.5, observed
inner NH4 central-interval coverage is only.61565 versus.82737 at alpha0.
These are model-member pooled diagnostic coverage rates, not independent
samples or a claim of nominal90-percent coverage. Preserve this limitation
even if a subsequent frequency gate passes.

support_audit.json and support_results.md quantify the observed-state event
support limitation on inner30 only, independently of outcome selection. At
alpha.5,9 of26 actual new DO events have no supporting member, even with
arbitrary within-support noise. This is not a free-running impossibility
claim or a new acceptance criterion. It does not change the ongoing search.

## Post-primary effect decomposition

The primary inner choice is pit_b95_a0.25 and passes the unchanged point gate.
It changes both bank calibration and alpha from e40's selected.5. After this
primary outcome, define a separate fixed2x2 diagnostic: original/matched bank
crossed with alpha.25/.5, all with block95 and identical weights/row keys.
decompose.py records this post-outcome timing and verifies both primary cells
reconstruct their existing metrics. Report all four cells and date-paired
Brier contrasts. Do not select a replacement alpha or bank from these cells;
the primary summary hash must remain unchanged. Full event audit is still
required and a point pass alone does not establish improved DO prediction.

```powershell
python Threemool/Lee/experiments/e51/decompose.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --freeze Threemool/Lee/experiments/e42/environment_freeze.json --runs results/e51_blended_pit --out Threemool/Lee/experiments/e51/decomposition.json
```

## Preflight implementation correction

The initial preflight assumption that every observed calibration state lies
inside the model's fitting-only clip bounds fails:14 of2850 observed NO3 states
exceed its original upper bound. All NH4/DO inputs are inside their bounds.
Do not change data, clip bounds or the adoption gate. Include original final
clipping when transforming conditional quantiles. NO3 quantiles are exactly
equal at all levels and its PIT stays.5, so interpolation and clipping commute
for that deterministic channel. Verify that equality and actual clipped-kernel
interpolation numerically; retain an explicit in-bounds assertion for safety
states where the stochastic argument requires it. This corrects a preflight
assumption in e51, not the parent model or a failed outcome criterion.
