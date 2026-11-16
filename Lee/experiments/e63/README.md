# e63: Risk minimization inside the existing hard action trust region

## Prospectively fixed hypothesis and scope

The completed e56 held-training-state proposal diagnostic finds that horizon4
risk-only choices change24.0954 to36.7881 percent of logged actions, while
the squared-action penalty permits changes on only.0611 to3.1614 percent.
Predicted gains are small and many curves are flat. This does not prove that
the unpenalized actions improve actual outcomes. Test that hypothesis rather
than choosing a smaller penalty by validation search.

Keep e56's primary horizon4 rate models, every original401280-step FCSRL
checkpoint, own captured training streams, observations, targets and e42
research environment unchanged. Fit no new regressor and add no PPO or
environment-training steps. Change only the controller score from summed
predicted NH4/DO mean rate plus squared action adjustment to summed rate alone.
Retain the hard candidate offsets[0,-.125,.125,-.25,.25], clipping to[-1,1]
and stable first-index ties, so a flat curve keeps the original action.
No candidate extends beyond the old.25 action radius before air matching.

Primary penalty coefficient is exactly0. Coefficient1 is an implementation
identity null for the original e56 controller, not another selectable primary.
No intermediate coefficient, alternative radius, horizon1 fallback, seed
selection, checkpoint selection or validation parameter search. Predictions
are learned mean rates, not guaranteed event probabilities or safety bounds.
Removing the penalty can exploit model errors, especially out of distribution.

This resumes agent-extension investigation on the already frozen e42 research
world after e58/e61 environment candidates fail. It neither replaces that
world with a failed candidate nor resolves its DO timing/ranking, season,
counterfactual and reused-validation limitations. Even an agent benefit here
is simulator-bound evidence and does not by itself complete the full user's
environment-safety objective or establish real-plant safety.

## Required checks before policy outcomes

Keep all legacy full-budget/checkpoint/source/freeze/model/seed guards. Verify
coefficient1 reproduces original e56 actions on all five held training-state
sets, and coefficient0 matches independent sklearn/rate/argmin actions.
Compare coefficient0 held choice counts with the already preserved proposal
diagnostic, without new environment queries or parameters selected from labels.
Check clipped duplicate candidates, stable ties, flat-risk identity, radius,
unchanged NN inputs and strict serialization/reload of only the fixed settings.
Reject invalid coefficients, changed model/config hashes and unexpected fields.

Before the complete study, require original native baseline replay, serialized
primary replay and an independent full-day primary action/trajectory replay
with the actual executed previous-air feedback. Independently audit both native
and matched primary paths in the complete study. Do not alter historical files
to support the new wrapper; use separate scoped adapters and metadata.

## Fixed comparison and adoption

Evaluate all five original final policy seeds0..4, each with five frozen model
members, eight draws and40 reused validation days. Match air with the original
quality-blind e44 selector, unchanged endpoints/ties/query budget and actual
per-seed1-percent tolerance. Apply its offset after the wrapper, preserving
native results and offset-dependent previous-air feedback. Never pick an
offset using quality. Do not change the energy/cost definitions.

Primary comparator is original frozen-environment FCSRL, using its same exact
baseline checkpoints and full budget. Secondary comparison to completed e56
horizon4 coefficient1 must verify same base/rate models and reference air; it
cannot rescue failure against FCSRL. Preserve every native/matched/query output
and report air, cost, NH4 and DO separately, sample SDddof1 and10000 paired
policy-seed bootstrap intervals. Model members/noise draws/dates are not policy
training seeds. Reused original seeds are not new independent training runs.

Keep the existing complete criterion: all five air matches, cost and NH4
difference upper endpoints below zero, DO upper endpoint at most zero. A DO-only
benefit, a favorable secondary contrast or more changed actions is insufficient.
Preserve all failures and uncertainty, do not lower criteria, never access the
final41 test days, and upload no source observations/captured arrays/banks.
Necessary weights already live in e55/e56 archives; record exact references
and preserve the new wrapper configuration, code, results and replay method.

The specification is committed at6b8df22 before implementation or new controller
trajectories. trust_policy.py now implements only coefficient0 primary and
coefficient1 identity null, the unchanged grid, stable ties and strict config
serialization. preflight.json passes all five held training-state sets,85120
rows per seed, independent sklearn action decisions, original coefficient1
identity and exact prior diagnostic choice counts. Radius, unchanged inputs,
flat-risk/clipped-boundary identity and config tamper/invalid-coefficient
rejection checks pass. No environment query or validation is performed here.

The initial preflight executes the uncommitted core working tree based on
6b8df22; preflight.json records exact executed source SHA256 values. The core
and its completed checks are committed before any new validation trajectory.
From the workspace, reproduce into a new destination:

```powershell
python Threemool/Lee/experiments/e63/preflight.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --captures results/e55_study/policies/new_ppolag_fcsrl --baselines results/e42_baselines/new_ppolag_fcsrl --study results/e56_study --out results/e63_held_preflight_replay.json
```

Use pinned threemool Python/CPU, DARROW_FAST_DYNAMICS=1 and OMP/OPENBLAS/MKL/
NUMBA thread counts1. Existing output files are rejected. Captured observation
arrays remain local. No preflight process remains live. Next implement the
guarded comparison and independent native/matched action audit, require the
full native-only smoke, then run the fixed all-five-seed matched-air study.

The comparison/audit implementation is committed ata337095 before any new
controller validation trajectory. The native-only full seed0 smoke launches:

```powershell
python Threemool/Lee/experiments/e63/compare_trust.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline results/e42_baselines/new_ppolag_fcsrl/seed_0 --candidate results/e55_study/policies/new_ppolag_fcsrl/seed_0 --correction results/e56_study/models/h4/seed_0 --out results/e63_native_smoke/seed_0 --native-only
python Threemool/Lee/experiments/e63/audit_trust.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline results/e42_baselines/new_ppolag_fcsrl/seed_0 --candidate results/e55_study/policies/new_ppolag_fcsrl/seed_0 --correction results/e56_study/models/h4/seed_0 --comparison results/e63_native_smoke/seed_0 --out results/e63_native_smoke/seed_0/action_audit.json --null-check
```

The first command verifies original native checkpoint replay and independent
base/model/config reload of the primary. The second starts only after the
first succeeds, independently rebuilds all152000 native choices using sklearn,
replays physical trajectories/executed-air accounting and checks full-day
flat-risk identity against original FCSRL. Its controls are implementation
checks, not additional performance seeds. Smoke console logs are
results/e63_smoke_compare.log and results/e63_smoke_audit.log. Actual air
matching is deliberately absent here and remains required in the complete
five-seed study and matched-action audits. The complete native smoke now passes,
including all152000 independently reconstructed choices, physical trajectories,
executed-air accounting and full flat-risk null replay. native_smoke.json and
native_smoke_audit.json preserve the completed compact records. Native-only
figures are not matched-air improvement evidence. Smoke handle56806 is terminal.

The complete study launches under sourcea5f0d7b after those smoke guards pass:

```powershell
python Threemool/Lee/experiments/e63/run_study.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --policies results/e55_study/policies/new_ppolag_fcsrl --baselines results/e42_baselines/new_ppolag_fcsrl --previous results/e56_study --smoke results/e63_native_smoke/seed_0 --out results/e63_study
```

Five fixed comparisons run first, then five independent native/matched action
audits, the unchanged e44 five-seed report and the same-base/model secondary
penalty contrast. All source files are pinned and checked throughout. Each
stage writes logs/status, failures stop later stages, and the runner rejects
an existing output directory. Complete reports stay under results/e63_study/
final_reports until verified. Parent console log is results/e63_study_run.log.
No full result is available yet. Do not duplicate live stages or change their
pinned runtime files. No model fitting, PPO retraining or final test is added.

After all stages finish, package the necessary unchanged checkpoints, h4 rate
models/provenance, fixed configs and complete generated outputs with:

```powershell
python Threemool/Lee/analysis/package_trust_artifacts.py --results results --study results/e63_study --policies results/e55_study/policies/new_ppolag_fcsrl --previous results/e56_study --smoke results/e63_native_smoke/seed_0 --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out results/artifacts/lee-e63-risk-only.zip --index Threemool/Lee/experiments/e63/artifacts.json
```

The packager requires terminal complete status, all five native/matched action
audits, exact model/config/checkpoint hashes and recomputed e44 report identity.
It excludes source observations, captured arrays and stochastic banks. Its
partial-study rejection is verified without creating an archive or index;
positive complete-study packaging remains pending. No partial archive is an
experiment result, and the existing source/runtime hashes remain untouched.
