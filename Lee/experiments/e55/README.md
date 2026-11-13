# Lee/e55: Learned one-step cost correction of an unchanged FCSRL policy

## Prospectively fixed hypothesis

e54's additional observations do not improve control despite e53's predictive
information gain. Test using a learned next-step cost prediction directly for
small action corrections. Keep the base11-input FCSRL policy and the e40
environment fixed in definition. This is an agent extension, not a replacement
environment, modified cost or proof of real-plant safety.

## Equal transition budget and passive data capture

Do not fit the correction model on the extra random episodes generated in e53.
Re-run the unchanged original FCSRL policy seeds0..4 for401280 transitions each
while passively recording the same public11 observations, executed clipped
action, terminal next observation and NH4/DO costs already seen by PPO. A
recording adapter must not change RNG, observations, costs, rewards, reset
sampling, member/noise paths, PPO/PID or representation targets. Keep the
original400000-step request,8 environments,760-step horizon and five evaluations.

The recorder adds no environment step or label query. Check initial action/RNG,
physical trajectory and final policy/representation tensor agreement with the
existing FCSRL references. If agreement fails, diagnose before interpreting
performance; do not silently choose a favorable reference. Generated captures
stay local and outside Git/archives. The candidate uses its own captured
training stream, with the same per-policy transition budget as its baseline.
Additional supervised compute is recorded and is not claimed to be free.

## Cost models fixed before outcomes

Use current public observations plus executed normalized action,12 inputs.
Set NO3 input to zero because the frozen safety transition excludes it. Fit
separate binary NH4 and DO classifiers. The labels remain NH4>4 and maxDO>2;
expected summed next-step cost is their probability sum, not a new CMDP cost.
No hidden member/noise, extra exogenous variables or future labels enter input.

Fit the first52 whole updates (316160 transitions) and reserve the last14
updates (85120 transitions) solely for diagnostics. Both subsets are from the
188 policy-training dates; neither includes validation40 or final test41.
No probability calibration, reweighting or hyperparameter selection uses the
held subset. Deploy the same fitted models, without refitting on the holdout.

Each label uses one HGB classifier:200 iterations, learning rate.05,15 leaves,
minimum40 samples per leaf,L2=1, no early stopping or class weights. Fixed
classifier seeds5500/5501 are distinct from the five policy training seeds.
If a fitting label has only one class, use that constant probability and retain
the seed. Do not invent positives or drop a rare-event seed.

Constrain NH4 probability nondecreasing in NH4 and nonincreasing in DO1/2/3,
previous air and current action; constrain DO oppositely. Remaining public
exogenous inputs are unconstrained. These signs reflect the frozen simulator,
not empirically established plant counterfactuals. Check action/state order
and predictor agreement with sklearn before evaluation. Any fast inference
must independently match the original probability/logit calculations.

## Fixed action correction

Let b be the original deterministic FCSRL action. Candidate actions are
clip(b+delta,-1,1), with deltas ordered[0,-.125,.125,-.25,.25]. Choose the
smallest predicted probability-sum plus(a-b)^2, with first occurrence breaking
ties. Thus unchanged b wins a flat-risk tie. There is no searched penalty,
radius or threshold. A constant-risk null predictor must return the original
policy exactly, including full-day replay and clipping endpoints.

The original policy network/normalizer/representation are not fine-tuned by
this correction. Save both original native validation and corrected final
validation, but use only the corrected final controller for the candidate's
primary report. The held training scores and correction parameters are not
selected by validation. Report correction frequency, predicted-risk differences
and clipping. A model can misrank actions or act myopically; these are failure
hypotheses, not reasons to tune parameters after seeing outcomes.

## Full verification and unchanged comparison

Check passive capture, exact transition counts, update/episode separation,
public feature construction, original checkpoint tensors, probability bounds,
monotonicity, fast/sklearn inference, null-policy identity, serialization and
an independent corrected-policy replay. Preserve baseline and correction model
hashes. Do not rewrite historical e42/e54 or their pinned sources.

Run all five full policy seeds. Report cost, NH4 and DO separately with the
existing final-policy evaluator and quality-blind air-offset matcher applied
after the corrected policy output: clip(corrected_action+offset,-1,1).
Keep the original endpoints, query/tie rules and per-seed1-percent actual air
tolerance. Retain native and adjusted outcomes and every query. Do not move
the offset inside the correction model to rescue an unfavorable result.

Use the unchanged five-seed criterion: all air matches pass, paired cost and
NH4 interval upper endpoints below zero, DO upper endpoint at most zero,
10000 seed resamples and sample SD ddof1. Extra classifier fits, model members,
noise draws and dates are not new policy seeds. Preserve all failures and
uncertainty, including e40 DO onset/ranking limits and reused-validation bias.

Keep raw data, captures and stochastic banks outside Git. Publish compact
code/reports under Lee/e55 and authorized model/log artifacts with verified
checksums in the existing private draft. Original source d882b91054d52c7a63a18a21f7db67bbbed82b63
and Apache license remain. No final test, team-protocol, Paper or other-member
changes. Continue the active objective if this fixed hypothesis fails.

Status: specified after the complete e54 failure, before implementation or
capture/model outcomes. No new policy job has started for e55.
