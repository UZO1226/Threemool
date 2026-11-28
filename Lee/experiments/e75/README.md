# e75: Validate a continuous next-DO predictor before shielding anchored actions

Latest outcome: all five fits and the independent raw/held audit complete.
Only seed3 passes all admission gates; the all-five condition fails.
No conditional controller is evaluated. See [results.md](results.md).
Keep the prospective model, split, gates and conditional plan below unchanged.

Prospective study after complete e74 DO failure,2026-11-17. e74 improves
cost/NH4 but its DO upper endpoint remains positive. Horizon4 rate heads have
wide zero-risk plateaus. Test whether predicting the next physical DO value
can exclude proposals near the actual2mg/L boundary that rate-only guards
miss. This is a hypothesis, not a causal explanation of the failed trial.

The new predictor is an agent-side auxiliary model learned from existing
frozen-simulator training captures. It does not replace e42, repair real
DARROW safety forecasts or use new sensor/future information at action time.
Preserve all prior failures, authenticated matched e63 and final test41.

## Fixed offline model and partition

Use each unchanged401280-step policy's own immutable captured transitions.
Input is obs11 plus its logged executed action,12 public features retaining
NO3. Target is max of next_obs DO1/DO2/DO3 in mg/L, including true terminal
next observations. No next observation enters inputs. Authenticate all
capture hashes, seed IDs, original/captured parity, source/data/freeze identities
and training-day membership before fitting.

Fit updates1 through40 (243200 rows), calibrate41 through52 (72960), and
diagnose53 through66 (85120). These are disjoint update blocks, not disjoint
dates or a base-NN holdout. All are official training data. Neither validation40
nor test41 contributes labels, thresholds, models or calibration margins.

Fit one HistGradientBoostingRegressor per actual policy seed with squared
error,200 iterations, learning rate.05,15 leaves, min_samples_leaf40, L2=1,
early_stopping=False, random_state7500. Reuse the existing DO direction
constraints: NH4 negative, DO1/2/3, previous air and proposed action positive;
other coordinates unconstrained. Clip mean prediction below at0 only.
Use original sklearn and an independently checked compiled raw predictor.
Do not clip this physical forecast to[0,1] or treat it as a probability.

The one-sided margin is max(0, empirical95th percentile of calibration
y minus clipped mean), using the higher order statistic. Upper forecast is
clipped mean plus that fixed margin. Also calibrate current-state persistence
with the identical rule. This empirical margin has no iid/conformal or
conditional off-policy coverage guarantee for correlated logged data.

Report mean MSE/MAE/bias, upper pinball loss and coverage, next-DO>2 false
positives/negatives, and new-onset performance for current maxDO<=2 followed
by next maxDO>2. Upper-forecast AP/AUC are ranking scores, not probability
calibration. Compare current-state persistence and the fitting-mean constant.
Use10000 paired held-training-date resamples, RNG7502, for MSE contrasts and
coverage uncertainty conditional on the fitted policy/model. Do not count
dates as independent policy seeds or mix the5 original policy seeds.

All five original policy seeds must independently satisfy these admission
gates before any candidate quality evaluation: exact provenance and row
counts, original/compiled raw predictions within1e-12, monotone action probes,
held mean-MSE difference interval upper endpoint below zero versus both
point controls, held upper coverage at least.95, and next-DO false negatives
no greater than calibrated persistence. Report onset misses and uncertainty
even when all gates pass. A failed seed cannot be dropped, pooled away or
rescued by weakening these conditions. Stop this candidate stage and retain
the model failure if admission fails.

The RL interaction budget remains401280 steps per policy with zero additional
environment-training transitions. This extension adds one200-iteration
offline supervised fit and calibration per seed. Report that added model,
parameters, CPU time and data reuse explicitly; do not claim equal total
optimizer work, model count or wall-clock compute with e63/e74.

## Conditional controller stage, only after model admission

Keep e74's authenticated matched reference, nine centered proposals, radius.5,
original horizon4 component guard, fitting mean-action target, two shadow
prices,64 bisections, clipping and first-column ties. Add only this fixed
admissibility condition: a proposal's upper next-DO forecast is at most
max(2, reference upper forecast), with numerical tolerance1e-12. The reference
always remains feasible. This allows safe-reference headroom up to the legal
threshold and requires nonincrease when the reference forecast is already
above it. Do not replace this max rule with an unattainable min rule.

Eligibility remains a strict admissible horizon4 summed-rate gain over
column0, not a new DO reward. Fit only eligible observations from updates1
through52, preserving their matched-reference mean and overall mean within
1e-10. Held observations fit nothing. Monotone action forecasts should keep
the interpolated proposal admissible; verify the actual interpolated action
with original sklearn on all held states/probes and executed trajectories.

The original air-only matcher still adds its offset after the wrapper.
The shield therefore does not constrain final matched actions. Report final
executed component risks and upper-DO threshold excesses separately, without
clamping/canceling or quality-selecting the matching offset. Do not claim
formal safety, repaired physics, counterfactual identification or zero DO.

Require e74's matched-reference identity and frozen-offset NN null controls,
plus shield-disabled identity with the old e74 parameters on all held rows
and full seed0 native trajectories. Run every original policy seed, all native
and matched outputs and independent audits. Primary remains matched e63;
e74 is a secondary mechanism contrast. All air matches and unchanged10000
paired policy-seed resamples with RNG2026 are mandatory: cost/NH4 upper
endpoints below zero, DO at most zero. No new criterion, favorable seed
selection, checkpoint selection or block pooling is allowed.

If the original block passes, repeat the unchanged entire method including
the fixed auxiliary fit/calibration on exposed seeds5 through9, with separate
criteria. If both development blocks pass, prepare declared actual seeds10
through14 for additional separate confirmation before adoption. Exposed
seeds are not unseen confirmation and reused validation40 stays exploratory.
