# Lee/e56: Four-step cost-rate correction with a one-step regression control

## Prospectively fixed hypothesis

e55 reduces total cost and DO violation at matched air but fails the complete
criterion because NH4 remains uncertain. Test whether a one-hour target can
capture slower NH4 consequences that a next15-minute target misses. This is
a hypothesis, not a proven explanation of e55. Preserve all e55 results and
the original criterion; do not tune its successful seeds, radius or penalty.

Use the same five unchanged11-input FCSRL policies and passively captured
401280-transition streams from e55. Full byte/config/training-record parity
with the original references is already established. No additional PPO run,
environment transition or model-oracle label is collected. Supervised fitting
and evaluation compute are additional. These remain five policy training
seeds0..4, not ten independent seeds or extra seeds for every classifier.

## Target and controlled change

Primary method: predict the average NH4 indicator and average DO indicator
over the next four executed transitions (one hour when four steps remain).
For each trajectory position, include its own transition cost and up to three
following costs, stopping at the95-step day boundary. Divide by the number
of available transitions. Never include an autoreset day's cost, another
environment lane, a later update or a future observation as an input.
This is an expected short-horizon violation rate, not the probability of any
violation in the window. Its MSE must not be mislabeled binary-event Brier.

Required control: the same regression implementation, features, model size,
constraints, fitting stream, seeds and controller, with horizon1. Thus the
primary/control difference is the label horizon. The e55 classifier is an
existing reference, not a sufficient loss-matched horizon control.

Both variants use current public11 observations and the executed normalized
action,12 features. Retain NO3, with no monotonic constraint: although it is
absent from the frozen safety transition, future behavior-policy actions can
depend on it. The horizon1 regression control retains it too, so feature and
loss changes versus e55 cannot be attributed to horizon alone.

Fit the first52 whole updates,316160 transitions, and use the last14 updates,
85120 transitions, for diagnostics only. Deploy the same models without
refitting on the holdout. Labels use only transitions within their own split
and day. The original captured behavior is stochastic and changes across PPO
updates; its future actions differ from final deterministic evaluation. These
are approximation and distribution-shift limitations, not causal plant labels.

Each label uses one HGB regressor, squared-error loss,200 iterations, learning
rate.05,15 leaves, minimum40 samples per leaf,L2=1, no early stopping or weights.
Use fixed regressor seeds5600/5601, distinct from policy seeds. Clip outputs
to[0,1]. Keep NH4 signs positive in NH4 and negative in DO1/2/3, previous air
and current action; use the opposite signs for DO. Other public features are
free. For future behavior-policy costs these are inductive assumptions, not
guaranteed feedback-trajectory order. Report held error and failures honestly.
Use constants for a constant fitting target and retain that policy seed.

## Fixed controller and verification

Use the unchanged final FCSRL deterministic action b and candidate offsets
ordered[0,-.125,.125,-.25,.25], clipped to[-1,1]. Score predicted NH4 rate plus
DO rate plus(a-b)^2. Keep penalty1 and stable first occurrence for ties. A
flat-risk null must recover the original policy exactly. Do not tune horizon,
penalty, radius, regressor loss or threshold using validation.

Before outcomes, independently verify horizon4 and horizon1 labels against
explicit within-day sums, terminal partial windows, lane/update/split boundaries
and future/reset exclusion. Require exact horizon1 identity with captured costs,
bounded finite targets, constant-target handling, serialized predictions,
public feature construction, fast/sklearn numeric agreement and the actual
fixed candidate choices. Preserve all original base checkpoints and sources.

Run both variants for all five policy seeds. Require original checkpoint,
config, budget, source/freeze identity and native replay guards, plus independent
corrected-controller replay. Apply the unchanged quality-blind air matcher
after correction, never inside it. Preserve native outcomes, every air query,
actual match error and all failed seeds.

Primary adoption criterion remains all five air matches within1 percent,
cost and NH4 paired-seed95 interval upper endpoints below zero, and DO upper
endpoint at most zero. Use10000 paired policy-seed resamples and sample SD
ddof1. The horizon1 regression control is not substituted if the primary fails.
Also report horizon4 minus horizon1 at the same common reference, checking
reference checkpoint and mutual air identity. An uncertain secondary contrast
prevents attributing a primary gain specifically to a longer horizon.

No final test, actual costs/rewards/thresholds, frozen environment, team protocol,
Paper or other member's files change. Preserve previous DO onset/ranking and
reused-validation limits. Do not claim real-plant safety or an independent
confirmation from these exploratory comparisons. Captures and original data
remain outside Git and archives; fitted controller models/logs and reproducible
source/summary hashes belong in the authorized draft artifacts.

Status: specified after complete e55 verification and before any e56 label
generation, fitting, smoke or performance outcome. No e56 job has started.
