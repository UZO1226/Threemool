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
