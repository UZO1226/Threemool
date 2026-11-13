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

## Implementation and smoke status

The prospective specification was committed at921e865 before implementation
or correction-model outcomes. Passive capture and fixed slow correction were
committed atdfee498. Subsequent inference, loading, audit and batch adapters
are recorded by their execution source hashes. No result changes the fixed
candidate list, penalty, fit/held update split, five policy seeds or criterion.

capture_check.json establishes190 vector steps (1520 transitions), two complete
episodes per environment, exact physical outputs and unchanged RNG. The
6080-transition original-policy smoke finishes with every final policy,
representation and normalizer file SHA256 exactly equal to the historical
e54 original-policy smoke. Its final native validation metrics are also exact.
capture_smoke_check.json records this limited result; it is not a full-budget
equivalence claim.

The smoke cost model uses only4560 fitting and1520 held transitions from that
one update, rather than pretending to have52/14 full updates. This special
partition verifies the pipeline only and is explicitly excluded from the
five-seed performance report. Held public features, terminal executed air,
physical threshold labels and saved metrics reconstruct exactly. Fast logits
equal sklearn and maximum probability discrepancy is1.12e-16; tested state
and action order violation is zero. Held-date uncertainty, new-event counts
and simple persistence/fit-prevalence controls are diagnostic, not tuning data.

compare.py verifies unchanged base checkpoint/config/source/budget identity,
replays its original native metrics and independently reloads/replays the
corrected policy. The smoke additionally checks full-day constant-risk null
identity and equality of every fast/sklearn correction decision. audit_correction.py
independently constructs public features and fixed candidate scores, preserving
all correction frequencies, clipping and predicted-risk differences.

The complete study is run from the workspace with the threemool Python
environment and one OMP, OPENBLAS, MKL and NUMBA thread per process:

```powershell
python Threemool/Lee/experiments/e55/run_study.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baselines results/e42_baselines/new_ppolag_fcsrl --out results/e55_study
```

The runner requires a completed smoke_pipeline.json, pins all runtime sources
and retains process logs/status. It trains all five captured original policies,
requires byte-identical full reference checkpoints before fitting, fits the
fixed classifiers, reconstructs held-stream diagnostics, evaluates both native
controllers and matched air, independently audits actual correction choices,
then invokes the unchanged e44 five-seed report. No extra environment training
transition is collected for a classifier. Supervised fitting and evaluation
compute are additional and separately recorded.

Capture files remain local under each policy's transitions directory. These
generated training arrays, original data and stochastic banks must be excluded
from artifacts. Re-running the identical collector reconstructs the training
stream; capture-file SHA256 values are recorded in fit_summary.json. Reproduction
must preserve source hashes, the e42 freeze and package versions. Final risk
model pickles are necessary controller artifacts alongside the base checkpoint;
the older policy-only archive helper does not include them automatically.

Status: smoke_pipeline.json passes all capture, checkpoint, held-model,
serialized replay, constant-risk full-day null,152000 fast/sklearn action and
independent public-feature/choice checks. The unchanged air matcher takes8
queries and reaches relative air error-.0001811442. No full policy seed is
complete at launch. The complete queued study starts under sourcebdd9c82 in
results/e55_study: five full captured-policy runs, then unchanged-reference
equivalence, fixed cost fits, held-stream model audits, matched-air comparisons,
independent action audits and the unchanged five-seed report. Do not duplicate
the queued jobs. Inspect actual handles and generated summaries rather than
inferring liveness from this note. Preserve pinned Python sources for replay.

All five full captured policies now finish and pass exact reference equivalence.
training_control_check.json verifies every final checkpoint file, all66 training
records and all five original native evaluation records per seed. Both cost
models per seed finish using316160 fitting and85120 held transitions. Independent
held-stream audits pass for all seeds; fast logits are exact, probability error
is at most2.23e-16 and tested coordinate-order violation is zero. The full native
and matched-air comparisons and all independent action audits finish. Same-air
cost and DO intervals are below zero, but the NH4 interval crosses zero, so the
complete criterion fails. comparison.json and results.md preserve this partial
benefit and failed full criterion. No e55 job remains queued. Predictive AP,
source parity and held-data checks do not establish a complete improvement.
Regenerate this compact check from the workspace using
Threemool/Lee/analysis/summarize_correction_training.py
with --study results/e55_study, --baselines results/e42_baselines/new_ppolag_fcsrl
and --out Threemool/Lee/experiments/e55/training_control_check.json.
