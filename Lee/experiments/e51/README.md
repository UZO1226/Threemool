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
alpha*current_state+(1-alpha)*quantile, then estimate PIT of actual next state
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

Status: plan specified after e50, before e51 implementation or evaluation.
