# e73: Componentwise predicted-rate guard for budget projection

Prospective candidate after e71's failed separate replication and e72's held
component diagnostic,2026-11-17. Summed-risk eligibility permits trading
predicted NH4 risk for a DO reduction. Test a guard that excludes either
predicted component increasing relative to the same-state e63 native action.
This hypothesis uses already exposed development evidence, not untouched
quality confirmation. Preserve every earlier failure and criterion.

Keep original final401280-step policies, horizon4 regressors, obs11, frozen
e42, nine offsets[0,-.125,.125,-.25,.25,-.375,.375,-.5,.5], radius.5, clipping
and first-candidate ties. Compute the original e63 native action by minimizing
the summed rates over the original first five candidates. A proposed action
is admissible only when each predicted rate is at most its reference rate
plus the existing numerical tolerance1e-12. Replace inadmissible candidate
columns with the e63 reference action and its summed rate before the existing
finite64-step mean-action calibration. The reference is always available.
The replacement is a hard exclusion, not an adjustable risk penalty.

Only states with a strictly lower admissible summed rate by more than1e-12
are eligible. Outside eligibility return e63 exactly. Calibrate eligible
training states from updates1 through52 to their e63 native mean action,
with the original bounds[-2^26,2^26],64 bisections, two shadow prices and convex
action interpolation. Require overall fitting mean error at most1e-10; no
eligible fitting rows means reference-only mode for that seed. Do not discard
seeds. Held updates53 through66 do not fit anything.

Each risk head is monotone in current proposed air conditional on its other
inputs. Interpolation between admissible endpoints should therefore remain
componentwise admissible; independently verify the actual interpolated action
with original sklearn predictions on held probes and all executed trajectories.
Do not assume this guard constrains true safety or recursive future outcomes.
Record all predicted increases and any numerical violation as a control failure.
Previous-air feedback, state shift and model misspecification remain unresolved.

The component guard applies to the interpolated wrapper action relative to
same-state e63 native, before the unchanged post-wrapper air-matching offset.
Check this wrapper property on states visited in both native and matched
trajectories. The final matched action can violate the predicted component
bound after its offset. Report those final-action predicted changes separately
against the same-state authenticated e63 action including its fixed offset;
do not claim the guard constrains matched actions or change the matcher to
force a favorable result. Actual matched quality remains the primary test.

Before evaluation require known-answer exclusion and interpolation controls,
constant-risk NN identity, reference-only e63 identity, strict reload and all
five held checks. Run every original policy seed0 through4 with unchanged
e44 air-only matching and independent native/matched audits. Primary e63
includes its authenticated fixed post-wrapper offset. Report native air and
quality, matched per-seed values and all failure outcomes.

Require every actual-air match, and unchanged10000 paired policy-seed
bootstrap resamples with RNG2026: cost and NH4 difference interval upper
endpoints below zero, DO at most zero. If the original block passes, repeat
the entire fixed method on the exposed seed5 through9 block, with separate
criteria and no pooling. These seeds no longer constitute unseen method
confirmation. If both development blocks pass, predeclare actual training
seeds10 through14 for an additional separate confirmation before adoption.
Do not tune after either block, select checkpoints/seeds, change offsets or
quality-select air adjustments. Final test41 remains unused. Reused
validation40 remains exploratory even after new policy-seed confirmation.

## Training-only implementation checks and execution

All original seeds0 through4 now pass mean calibration and all85120 held
rows with zero predicted NH4 or DO increases in the pre-offset wrapper.
Independent1024-row original-sklearn decisions, actual interpolated-rate
checks, original e63 fallback, strict reload and e71 changed-row controls pass.
preflight.json records exact parameters and source identities. No policy
quality improvement follows from these training-only results.

The initial native smoke now reproduces its authenticated reference and
strictly reloaded candidate. native_smoke_audit.json independently verifies
all152000 actions and trajectories, zero wrapper component increases, fitted
flat-risk NN identity and reference-only e63 identity over complete native
trajectories. All five original-seed actual-air comparisons are now launched,
with separate native/matched independent audits still required. These controls
are not a completed five-seed improvement result.

Use the threemool environment, numerical threads1 and frozen dependencies:

```powershell
python Threemool/Lee/experiments/e73/check_guard.py --out results/e73_guard_check.json
python Threemool/Lee/experiments/e73/fit_guard.py --workspace C:/Users/user/Downloads/IoT_project2 --certificate Threemool/Lee/experiments/e64/baseline_authentication.json --out results/e73_guard_fit
python Threemool/Lee/experiments/e73/compare.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --candidate results/e55_study/policies/new_ppolag_fcsrl/seed_0 --baseline results/e42_baselines/new_ppolag_fcsrl/seed_0 --correction results/e56_study/models/h4/seed_0 --certificate Threemool/Lee/experiments/e64/baseline_authentication.json --preflight results/e73_guard_fit/preflight.json --projection results/e73_guard_fit/seed_0 --previous results/e63_study --out results/e73_native_smoke --native-only
python Threemool/Lee/experiments/e73/run_study.py --workspace C:/Users/user/Downloads/IoT_project2
```

Do not overwrite completed output directories; reproduce in a fresh workspace
with consistent new runner paths. The fitter reads the committed guard_check.json
certificate; if reproducing changed sources, regenerate and review that control
certificate before fitting. Source/model/data mismatches fail closed.

After the complete original five-seed study, independently recompute its
recorded episode aggregates, air-only choices and paired policy-seed intervals:

```powershell
python Threemool/Lee/analysis/verify_e71_blocks.py --trial e73 --original-only --workspace C:/Users/user/Downloads/IoT_project2 --out results/e73_initial_verified.json
```

The verifier retains its historical filename and unchanged default e71
two-block behavior. e73 currently requires explicit original-only scope;
this cannot verify the exposed5 through9 or newly declared10 through14 block,
and its original-only success flag cannot claim two-block success. Regression
checks preserve e71's original pass and fresh failure under identical criteria.
