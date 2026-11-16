# e60: Fixed feedback horizon and temporal-noise diagnosis

This is a calibration-only mechanism diagnostic after completed e58/e59.
No new model fitting, validation evaluation, final test, policy training,
candidate adoption or parameter selection. Preserve the e42 freeze and all
failed results. The calibration dates also estimated the PIT banks, so these
are descriptive comparisons, not held-out confirmation.

Fix both e51 and e58 at their existingpit_b95_a0.25, five members, eight draws
and30 calibration dates. Cross reset intervals1/4/16/95 with original95-step
PIT paths and a deterministic within-day permutation of those same paths.
Preserve complete five-channel tuples, every daily multiset and member/draw/
date identity. Use independent SHA256 keys e60shuffle:day:member:draw solely
for the permutation. Do not select a shuffle, horizon, seed or model from the
result. Reset current state to the factual observation at step0,H,2H,...;
retain original factual exogenous inputs and current/previous air throughout.
This access to intermediate actual states is diagnostic, not deployment.

Before interpreting outcomes, require reset95/original scores to reconstruct
both stored selected inner summaries, and reset1/original NH4/DO Brier/AP/AUC
to reproduce e59 calibration observed-state results. Require exact tuple
permutation/inverse and finite bounded paths. Report all16 conditions,
NH4/DO frequency/Brier/AP/AUC/onset0/1/4, normalized and per-state MAE, and
paired circular five-day date-bootstrap Brier differences for shuffled minus
original at each reset interval. These are reused calibration comparisons;
model members and noise draws are not policy seeds.

Failure of an identity control is an implementation problem to fix, not a
reason to loosen the earlier adoption criterion. Retain failures and source
hashes. Use these results to motivate a later prospectively fixed model
change; never rescue e58 with a favorable diagnostic condition. Publish
aggregate summaries only, excluding source/noise/trajectory arrays.

The specification and implementation are committed at2e4bd6d before outcomes.
The diagnostic launches in results/e60_feedback_diagnosis with:

```powershell
python Threemool/Lee/experiments/e60/diagnose_feedback.py --data kaggle_data/darrow_15min.pkl --e51 results/e51_blended_pit --e58 results/e58_conditional_upper --e59 results/e59_context_diagnosis.json --out results/e60_feedback_diagnosis
```

Use the pinned threemool environment, CPU, DARROW_FAST_DYNAMICS=1 and all
OMP/OPENBLAS/MKL/NUMBA numerical thread counts1. Two independent inference
tasks run concurrently; no fitting or new policy seed is added. Each finished
model/horizon writes a separate aggregate JSON. summary.json exists only after
all eight paired tasks and both identity controls pass. An existing output
folder is rejected. Inspect actual process liveness before resuming and never
duplicate a live diagnostic. No completed outcome is recorded yet.
