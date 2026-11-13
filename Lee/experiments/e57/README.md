# Lee/e57: Fixed executed-air smoothing of the unchanged FCSRL policy

## Prospectively fixed hypothesis

e55 gives a partial cost/DO benefit, while both e56 rate horizons fail complete
adoption. Test a distinct actuator mechanism without refitting a logged-cost
predictor: smooth the original policy's command toward the previously executed
air. The mechanism might reduce abrupt oxygen excursions or might worsen NH4
through lag. Neither outcome is assumed. The held proposal diagnostic does
not justify retuning a risk penalty or establish that risk-only actions help.

Use exactly the five original full401280-step FCSRL policies, seeds0..4,
with the same frozen e40 environment and original references. No PPO learning,
new supervised fit, environment-training transition or extra exogenous input
is added. This remains five policy training seeds, not new seeds for every
evaluation. Read previous air only from the public observation field10,
which the original policy already observes. Preserve original NN weights,
normalizer, representation, physical costs, rewards and thresholds.

## Fixed controller

Let b be the original clipped deterministic normalized action. Convert current
observed previous air to normalized units as p=clip(2*obs[10]-1,-1,1).
The primary action is clip(.5*b+.5*p,-1,1). Gain.5 is fixed before any outcome;
there is no searched gain, radius, penalty, forecast model or safety threshold.
This filter gain is separate from the unchanged e40 dynamics mixing alpha.5.

The initial previous-air observation is the same recorded initialization
available to the baseline. It can affect the smoothing transient; do not
claim that any result isolates a steady-state effect from initialization.
Clip only p inside the filter, including an initial measured air outside the
controller's command range; do not clip or alter the observation given to
the original NN. Subsequent p reflects the actually executed clipped action,
including any matching offset, never an unexecuted proposed action.

Gain1 is an identity null, returning the original action values and dtype.
It must reproduce original full-day native metrics and source/checkpoint
guards, including reset/terminal cases. It is a verification control, not
another independently trained performance method. The primary remains gain.5.

## Verification and comparison

Before primary outcomes, independently verify the public previous-air unit
conversion, clipping, convex combination, bounds and deterministic identity
on synthetic boundary/out-of-range observations. Check serialization/reload,
actual executed previous-air feedback, resets and terminal observations on
real frozen-environment rollouts. Require the full native gain1 null replay
and independent primary replay, with unchanged final model/source/config/
budget/freeze guards. Do not edit pinned historical Python sources.

Evaluate all five final-policy seeds with the same common five environment
members, eight noise draws and40 reused validation dates. Apply the original
quality-blind air matcher after filtering: clip(filtered_action+offset,-1,1).
Preserve its endpoints, tie/query rules and actual per-seed1-percent air
criterion. This changes the executed previous-air feedback and must not be
replaced by filtering an already offset base action.

Report native/matched air, total cost, NH4 and DO independently, sample SD
ddof1 and10000 paired policy-seed resamples. The unchanged adoption criterion
requires all five air matches, cost and NH4 upper interval endpoints below
zero and DO upper endpoint at most zero. Reduced command variation alone
is not adoption, and an unfavorable seed is not dropped.

As a secondary mechanism diagnostic, report adjacent executed-command changes
separately from the initial jump. This is not measured electrical energy or
plant counterfactual evidence. A state-local convex combination does not
guarantee globally smaller closed-loop changes after NN feedback and matching.
Keep known e40 DO onset/ranking, season, surrogate and reused-validation
limitations. No final test, team protocol, Paper or other-member change.
Preserve all failures and necessary original base/config/log/evaluation
artifacts without uploading source observations, captures or stochastic banks.

The specification was committed at390659c after complete e56 verification,
before e57 implementation, smoke or performance outcomes. air_filter.py now
implements only the fixed gain.5 primary and gain1 identity null. preflight.json
passes synthetic units/bounds/unchanged-NN-input/dtype/serialization checks,
190-step identity rollouts with nonzero offset, primary executed-air feedback
and reset/terminal checks, and original full native gain1 replay. Final base
files remain unchanged. These diagnostic queries add no training transition.
No gain.5 validation performance has been evaluated. Next implement the
source/config-guarded paired comparison and independent replay, then run all
five seeds with the unchanged matching/quality criterion. No e57 job is live.
