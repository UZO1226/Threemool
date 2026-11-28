# e69: Training-state air-budget projection

Status: complete, all five independent audits pass, joint quality criterion
fails. See results.md and comparison.json. Preserve the following prospective
specification and preflight history; it is not an adoption or fresh replication.

Prospective hypothesis, 2026-11-17. e67's wider zero-penalty correction raises
native air substantially and fails cost/NH4 at matched air, despite lower DO.
Test risk-guided allocation that first conserves e63's mean native action
on fitting states, instead of increasing air and relying entirely on a
subsequent uniform offset. This is a new fixed method, not a rescue of e67.

Reuse the unchanged e63 NNs and horizon4 risk regressors with e67's nine
ordered offsets0,-.125,.125,-.25,.25,-.375,.375,-.5,.5. For each state and
multiplier lambda choose the first minimum of summed predicted rates plus
lambda times the proposed action change from the NN. Fit only the mean
air constraint on the316160 captured states from updates1 through52.
The target is mean unoffset e63 radius0.25 action on those same states.
Do not use held updates, validation quality, native validation air, or e63's
validation-matched offset to choose lambda. Final comparison still uses
the authenticated matched e63 controller as primary, including its offset.

If lambda0 already matches the target mean within1e-12, retain lambda0.
Otherwise bisect the monotone mean-action curve between fixed endpoints
[-2^26,2^26] for64 iterations. Keep its two final bracketing multipliers.
Because a discrete grid can jump over the desired mean, use one global
convex interpolation weight between those two deterministic action policies
to reproduce the fitting-state target within1e-10. Store both multipliers
and the weight. The executed action can lie between grid actions; it remains
bounded and within radius0.5. No hyperparameter search or quality-selected
multiplier is allowed. Bisection uses mean action only, with the models and
target fixed. Predicted risk may be nonlinear under action interpolation;
do not claim an optimal projection, cost bound or causal guarantee.

Evaluate on85120 held training states and independent sklearn subsets before
any candidate environment comparison. Verify exact fit mean constraint,
bounded/radius-preserving actions, strict serialization and unchanged base
and risk weights. Verify lambda0 identity against e67 on all held states.
A constant-risk model with recalibrated lambda0 must reproduce the NN;
do not claim constant-risk identity with a nonzero fitted multiplier.
Held states are excluded from auxiliary calibration, not from PPO training.

Then run seed0 native serialization replay and independent full-day audit,
followed by all five original policy-seed native/matched comparisons and
independent sklearn action/trajectory audits. Use the unchanged e44 air-only
matcher against each actual e63 matched reference volume. Equal mean on
logged fitting states does not guarantee equal air in closed-loop rollouts.
Always retain actual native and matched air/quality. e67 is only a secondary
mechanism comparison, not the primary reference or an adoption fallback.

Keep the original401280-step environment training budgets, existing inputs,
e42 freeze and zero new environment-training steps. All five actual-air
matches must pass; cost and NH4 difference95% interval upper endpoints must
be below zero and DO at most zero versus e63,10000 paired policy-seed
bootstrap resamples, RNG2026. Do not change any rule after observing results.
Reused validation40 remains exploratory and test41 is excluded. Fresh
seed5~9 baseline preparation is separate; no fresh candidate quality has
been examined. Preserve both required blocks and all unsuccessful trials.

## Completed preflight and execution

All five calibration/held checks pass. Maximum fitting mean-action error
is2.22e-16. Held action changes are46100,42069,51825,50680,45128 of85120
for seeds0 through4. Independently rebuilt sklearn choices and serialized
reloads match exactly on the fixed1024-row subsets. Zero-multiplier decisions
reproduce e67 on all held rows. projection_check.json binds the executed
known-optimum, multiplier-sign and flat/zero controls to their source hash.
These are implementation and geometry checks, not improved quality.

The held candidate-minus-e63 native mean actions are not forced to zero;
for example, seed1 differs by about.026339 normalized action despite an
exact fitting constraint. Preserve this distribution/feedback limitation.
The complete native/matched study, not the logged mean constraint, decides
whether any actual air/quality improvement exists.

From the workspace with threemool Python and numerical-library threads1:

```powershell
python Threemool/Lee/experiments/e69/fit_projection.py --workspace C:/Users/user/Downloads/IoT_project2 --certificate Threemool/Lee/experiments/e64/baseline_authentication.json --out results/e69_projection_fit
python Threemool/Lee/experiments/e69/run_study.py --workspace C:/Users/user/Downloads/IoT_project2
```

Before run_study.py, compare.py must finish the seed0 native-only smoke into
results/e69_native_smoke using its projection seed0 config, original e55 NN,
e56 h4 models, e64 certificate, fitting preflight and e63 study. The runner
then requires its independent full trajectory, zero-multiplier flat-risk
null and full e67 identity controls before starting all five comparisons.
Full per-seed audits precede the unchanged e44 report. Use separate unused
workspace/output paths for reproduction; completed artifacts are not replaced.
At this snapshot, native serialization replay passes and its independent
smoke audit is running. No policy gain is established.
