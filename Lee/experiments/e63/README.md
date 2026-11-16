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

Status: specified after inspecting completed e56 held-state evidence and
existing implementation guards, before e63 implementation, preflight or any
new controller trajectory. Next implement the guarded wrapper and held-state
identity checks, then a full native-only smoke before all-seed matching.
