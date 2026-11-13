# DARROW follow-up reproduction

These are exploratory environment experiments. None of e26 to e39 passes all
adoption criteria. e40 passes the original point-estimate gates and independent
reconstruction, with the event timing and uncertainty limits in its results.md.
Environment-member seeds are not policy-training seeds.
The official final test41 remains unused. Repeated validation40 is not an
independent confirmation set.

## Runtime and source data

Recorded runtime: Python3.11.15, numpy1.26.4, pandas2.2.3, sklearn1.9.1,
torch2.15.0 CPU and numba0.68.1. Use the original experiment READMEs for exact
settings and source commits. e40 pins sklearn and checks the original leaf
update function's source hash before applying its temporary wrapper. No
installed package files are changed.

Obtain the DARROW files through their authorized distribution route. Raw data
are excluded from Git and release attachments. Original input SHA256 values:

| File | SHA256 |
| --- | --- |
| train_val_test_online_dataset.csv | 0a8d06be567d012f5d3791763c9d910c83aac4787d5183a7a505c9b4fc7c9a82 |
| dynamic_influent.csv | 8c22eba0527371792511af966d52641ddf35209f6b25394da387fccd0380c81e |

The existing prepared input is kaggle_data/darrow_15min.pkl. Training uses
188 dates, inner selection normally reserves the last30 training dates, and
validation uses40 dates. e28 explicitly changes only the inner partition.
Do not substitute evaluation files or tune on the final41 dates.

Legacy prepared zone_fractions metadata uses medians from the full table.
Current recorded-air experiments use the actual zone arrays directly. The
train and validation total air never falls below the fallback threshold, so
that metadata did not affect these runs. Before a future policy adapter, derive
fallback fractions from official training dates only. Preserve the canonical
input and historical outputs rather than silently rewriting them.

## Saved checkpoints and logs

Follow-up draft release tag: lee-e26-e30-followup-20261112, release id442901392,
repository UZO1226/Threemool. The tag and id identify the draft reliably;
GitHub can change its temporary untagged HTML URL when release notes change.
Draft access requires an authorized repository account.

| Archive | Manifest |
| --- | --- |
| lee-e26-e30-followup.zip | [e26](../experiments/e26/artifacts.json) |
| lee-e31-e33-followup.zip | [e31](../experiments/e31/artifacts.json) |
| lee-e34-e37-followup.zip | [e34](../experiments/e34/artifacts.json) |
| lee-e38-e39-followup.zip | [e38](../experiments/e38/artifacts.json) |
| lee-e40-frozen-environment.zip | [e40](../experiments/e40/artifacts.json) |

Download with an authorized GitHub CLI session, for example:

```powershell
gh release download lee-e26-e30-followup-20261112 --repo UZO1226/Threemool --pattern lee-e38-e39-followup.zip --dir results/release_artifacts
Get-FileHash results/release_artifacts/lee-e38-e39-followup.zip -Algorithm SHA256
```

Check archive SHA256 against its linked manifest before extraction. Each zip
contains artifact_manifest.json with individual file hashes. Extract folders
under results/ to recover the documented run layout. Do not load pickle or
Torch files from untrusted sources.

Archives contain fitted weights and full JSON/log outputs, excluding raw
observations, residual/PIT banks and downloaded publications. Those banks can
be regenerated using the original data and scripts. The generic audits expect
them locally, so a downloaded checkpoint alone is not a complete audit input.
Re-running a full experiment into a new output folder regenerates them.

The original comparison model is expected at kaggle_data/darrow_surrogate_gbm.pkl.
The earlier approved e14/e15 archive stores it as environment/old_delta_gbm.pkl;
copy it to the expected path only after verifying that archive's manifest.
e16 also generates the persistence baseline. Paths stored in historical JSON
may be absolute Windows paths and need an explicit local-path adjustment when
running on another machine. Do not edit the recorded historical summaries.

For the frozen e40 environment, e42/rebuild_bank.py regenerates the excluded
PIT bank from authorized input and the five downloaded quantile checkpoints.
It rejects an existing output and requires the regenerated SHA256 to match
environment_freeze.json. A separate output results/e42_rebuilt_pit.npz was
tested to reproduce the frozen bank exactly. Restore the bank to its frozen
path before running FrozenDynamics. e42 also corrects terminal reward/air
accounting for both old-model and new-model policies; historical policies
remain separate sensitivity references.

## Execution and independent audits

Run from the workspace above Threemool, set OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to1, and use a fresh output directory.
The runners reject an existing directory to prevent overwriting prior runs.
The e40 README gives both the synthetic regression and full run commands.

For neural runs, e24/audit.py reconstructs selected checkpoints, verifies their
hashes, compares independent Torch and NumPy full-day predictions, diagnoses
observed-state resets and reports circular five-day date-bootstrap intervals.
Select the correct model kind:

| Run | model-kind | Special input |
| --- | --- | --- |
| e27, e29, e30, e31 | slow | e29 uses its saved continuation summary |
| e28 | base | Use e28's partition reference as documented |
| e32, e34 | context | Fixed initial episode context |
| e33 | kinetic | Temperature-scaled reaction prior |
| e36 | context | parent-runs points to e34 weights and residual bank |
| e39 | monod | Nonlinear implicit NH4 reaction |

e35/e37 assimilation and e38/e40 quantile forecasts have their own runners and
embedded checks. Do not load a context or kinetic checkpoint into the base
class merely because its parameter names fit. Quantile inference does not
require the training-time leaf wrapper after the trees have been saved.

Before pushing the personal branch, run scripts/check_scope.py with its branch
name and scripts/build_paper.py --check from Threemool. Keep changes under Lee/,
preserve final test41 and all failed runs, and do not push directly to main.

## Frozen-environment policy comparisons

The separate draft release id443333191, tag
lee-e42-e44-policy-comparison-20261113, contains completed policy subsets and
generated simulation evaluations. Draft assets require repository access;
their presence does not establish an improvement. Stable release id and tag
are recorded because GitHub may change its untagged draft URL after edits.

| Archive | Index with verified remote size and SHA256 |
| --- | --- |
| lee-e42-original-model-baselines.zip | [Five original-model FCSRL policies](../experiments/e42/old_baseline_artifacts.json) |
| lee-e42-frozen-model-fcsrl-baselines.zip | [Five frozen-model FCSRL policies](../experiments/e42/new_fcsrl_artifacts.json) |
| lee-e42-frozen-model-ppo-baselines.zip | [Five frozen-model PPO policies](../experiments/e42/new_ppo_artifacts.json) |
| lee-e43-split-five-seed-policies.zip | [Five split policies](../experiments/e43/split_artifacts.json) |
| lee-e43-capacity-five-seed-policies.zip | [Five capacity-control policies](../experiments/e43/capacity_artifacts.json) |
| lee-e44-split-five-seed-evaluations.zip | [All split air queries and evaluation cases](../experiments/e44/split_evaluation_artifacts.json) |
| lee-e44-capacity-five-seed-evaluations.zip | [All capacity air queries and evaluation cases](../experiments/e44/capacity_evaluation_artifacts.json) |
| lee-e45-severity-five-seed-policies.zip | [Five severity policies](../experiments/e45/artifacts.json) |
| lee-e45-severity-five-seed-evaluations.zip | [All severity air queries and evaluation cases](../experiments/e45/evaluation_artifacts.json) |
| lee-e46-cold-five-seed-policies.zip | [Five cold-sampling policies](../experiments/e46/artifacts.json) |
| lee-e46-cold-five-seed-evaluations.zip | [All cold-sampling air queries and evaluation cases](../experiments/e46/evaluation_artifacts.json) |
| lee-e47-ppo-five-seed-diagnostic.zip | [All equal-air PPO diagnostic cases and queries](../experiments/e47/evaluation_artifacts.json) |
| lee-e48-finite-day-five-seed-policies.zip | [Five finite-day auxiliary policies](../experiments/e48/artifacts.json) |
| lee-e48-finite-day-five-seed-evaluations.zip | [Five completed evaluations and preserved interrupted outputs](../experiments/e48/evaluation_artifacts.json) |

The separate environment follow-up draft id442901392 also contains
lee-e41-long-context.zip. Its [artifact index](../experiments/e41/artifacts.json)
verifies remote size/SHA256 for all checkpoints and the complete training log.
The residual bank is excluded and regenerated by the e41 training command.
The candidate fails adoption; see [full result](../experiments/e41/results.md).

The same environment draft includes lee-e49-next-step-diagnostic.zip, indexed
in [e49 artifacts](../experiments/e49/artifacts.json). It contains the20
diagnostic classifiers and full log, excluding raw observations and probability
arrays. Re-run e49/diagnose.py to recreate the arrays, then e49/audit.py to
independently reconstruct probabilities and metrics. This is a diagnostic,
not a replacement environment or a policy improvement.

The environment draft also contains lee-e50-quantile-exposure.zip with all
quantile checkpoints and complete console log. Its [index](../experiments/e50/artifacts.json)
verifies remote size/SHA256. Re-run e50/exposure.py to regenerate its excluded
PIT bank. Both independent audit commands are in the e50 README. This candidate
fails adoption despite lower MAE; preserve its failed event predictions.

lee-e51-blended-pit.zip in the same environment draft preserves identical
copied e40 checkpoints, calibration/decomposition console logs and compact
runtime summaries. [Its index](../experiments/e51/artifacts.json) verifies remote
size/SHA256. The selected PIT bank is excluded; e51/calibrate_blend.py recreates
it. e51 passes the point gate but does not improve DO probability prediction;
do not interpret this archive as adoption or an FCSRL extension result.

lee-e52-tail-innovation.zip preserves the fixed-weight tail candidate and full
console log, indexed in [e52 artifacts](../experiments/e52/artifacts.json).
The paired PIT/tail bank is excluded and recreated by calibrate_tail.py;
use the e52-specific auditor rather than the PIT-only auditor. It fails the
joint gate. lee-e53-history-full-range-probe.zip contains the30 training-only
diagnostic classifiers and log, indexed in [e53 artifacts](../experiments/e53/artifacts.json).
Generated arrays are excluded and recreated by probe_history.py. Neither
archive establishes a policy improvement. Both remote sizes/SHA256 verify.

The fixed lower-air e53 follow-up has a separate verified archive,
lee-e53-history-low-air-probe.zip, indexed in
[low-air artifacts](../experiments/e53/low_air_artifacts.json). Regenerate
excluded arrays with probe_low_air.py and verify with audit_probe.py. Preserve
both action-range reports; neither is a full FCSRL policy comparison.

The policy draft id443333191 contains four e54 archives, all with verified
remote size/SHA256: [history policies](../experiments/e54/history_artifacts.json),
[repeated-current policies](../experiments/e54/repeated_artifacts.json),
[history evaluations](../experiments/e54/history_evaluation_artifacts.json) and
[repeated-current evaluations](../experiments/e54/repeated_evaluation_artifacts.json).
These preserve every full seed, intermediate/final checkpoints and all air-only
queries. Both complete methods fail the unchanged criterion. The e54 README
records source, commands, native replay and primary/secondary reports.

Policy archives preserve results-relative run folders, final and diagnostic
checkpoints, logs, settings, source manifests and the original Apache license.
Evaluation archives instead contain seed_0 through seed_4, including native,
matched and every intermediate air-query JSON, plus five_seed_report.json.
Those rows are generated simulator metrics by date/member/noise draw, not
original plant observations. Neither archive type includes source data or
calibration banks. Verify archive and internal hashes before loading weights.

Restore the environment and excluded PIT bank first. e42/train_policy.py,
e43/train_split.py and e45/train_severity.py then use the same frozen adapter,
reward, cost limit12, corrected terminal air accounting and401280 transitions.
Full commands and runtime provenance are in each experiment README and run
manifest. Restore exact source files from the corresponding implementation
commit when replaying a historical run: checkpoint loading verifies its
training-source hashes, including all Python files pinned by that runner.
CRLF and LF text identities are handled explicitly, while binary hashes must
match exactly. Do not weaken source checks to accommodate unrelated edits.

For final-policy comparisons, e44/evaluate_pair.py uses common five-member,
eight-noise-draw and40-date validation cases. It verifies native replay,
actual training budget and checkpoint/freeze identity. Its air-only selector
adjusts candidate actions without reading water quality. Report both native
and adjusted metrics, using exactly five policy seed pairs in e44/report.py.
e44/capacity_contrast.py additionally checks that split and capacity variants
share the same original FCSRL checkpoint references and mutually matched air.
Do not pool1600 simulator cases as1600 policy seeds or select a favorable
intermediate checkpoint. e43 fails the prespecified improvement criterion;
retain both complete reports and their uncertainty intervals.

e45 is a separate severity-supervision candidate. All five full seeds and
matched-air comparisons finish and fail the improvement criterion. Its short
smoke/replay and unit checks are not additional policy performance seeds.
e46 completes a distinct colder-training-day sampling intervention, with a
DO-only benefit but a failed full criterion. Its checks and short smoke are
not additional performance seeds. e47 completes its existing-PPO matched-air
diagnostic, not a new extension. e48 finishes five full daily-horizon auxiliary
policies and all five matched-air comparisons, failing the full improvement
criterion. Its interrupted evaluation outputs remain archived; the retry
matches all144105 overlapping numeric values exactly without retraining.
Inspect actual processes before inferring whether a recorded run is active.

## e55 unchanged-policy one-step correction

The fixed specification predates outcomes at921e865. The complete implementation
and smoke pipeline are committed atbdd9c82. See the [e55 recipe](../experiments/e55/README.md)
and [smoke evidence](../experiments/e55/smoke_pipeline.json). The latter verifies
only one6080-transition policy, not five full-policy outcomes.

From the workspace root, with the pinned threemool Python environment and one
OMP, OPENBLAS, MKL and NUMBA thread per process, the complete command is:

```powershell
python Threemool/Lee/experiments/e55/run_study.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baselines results/e42_baselines/new_ppolag_fcsrl --out results/e55_study
```

This exact study is already queued. Inspect its live process and study_status.json
before doing anything; do not duplicate it. It keeps all five original11-input
PPO/FCSRL policies and budgets, passively records their existing transitions,
requires exact original checkpoint/config identity, fits NH4/DO cost classifiers
on the first52 updates and diagnoses the last14. Classifier seeds5500/5501 are
not additional policy seeds. Every final controller is evaluated natively,
after quality-blind air matching, and through an independent action audit.
The original e44 five-policy-seed bootstrap criterion remains unchanged.

Generated captures are stored in each policy's transitions directory and are
excluded from Git and archives. Their file hashes stay in fit_summary.json.
To recreate them, re-run the same original collector with train_stream.py;
the full base-equivalence check must pass before interpreting fitted models.
Neither held training metrics nor validation outcomes select correction
parameters. The extra supervised fitting and evaluation compute is recorded.

Artifacts must include both model_final and correction/risk_models.pkl. The
older policy-only packager silently omits pickle files, so use the dedicated
package_correction_artifacts.py only after the entire study completes. It
preserves all five base policies, correction models, logs, diagnostic/source
manifests and generated comparisons, with file/archive hashes and the original
FCSRL license. It explicitly excludes captures, source observations and banks.
The completed [e55 artifact index](../experiments/e55/artifacts.json) verifies
lee-e55-one-step-correction.zip in policy draft443333191, asset743367629.
It includes all five base checkpoints and fitted risk-model pickles, source
manifests, complete logs and all generated evaluations/audits. Original data,
captured arrays and stochastic banks are excluded. All five comparisons finish;
cost and DO components improve, but NH4 uncertainty makes the complete
criterion fail. Preserve this partial result rather than calling it adoption.

Historical e55 loaders verify recorded source paths. On a relocated checkout,
use the [portable source adapter](replay_correction_portable.py) to resolve
historical names to the actual current repository files while preserving exact
hashes. Do not edit stored manifests or relax the original guard. For example,
after restoring a completed study and its matching reference artifacts:

```powershell
python Threemool/Lee/analysis/replay_correction_portable.py --operation compare --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline results/e42_baselines/new_ppolag_fcsrl/seed_0 --candidate results/e55_study/policies/new_ppolag_fcsrl/seed_0 --correction results/e55_study/policies/new_ppolag_fcsrl/seed_0/correction --out results/e55_relocated_replay/seed_0
```

The new output path must not already exist. --operation action-audit exposes
the corresponding original independent action audit. Each operation writes
separate adapter provenance. Original source, checkpoint, configuration,
budget, freeze and native replay checks remain required. Replaying controllers
does not require captured training arrays; rechecking held fitting diagnostics
does require regenerating those excluded captures. Final test41 stays unused.

[The adapter check](portable_correction_check.json) covers synthetic missing
historical paths, an existing stale checkout, rejection of changed hashes and
repository traversal, plus exact152000-action smoke-audit replay. This is a
Windows reproduction check, not a full-budget performance result or evidence
that every operating system has been tested.
