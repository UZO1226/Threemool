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
not additional performance seeds. e47 next reuses existing PPO policies for
a matched-air diagnostic, not a new extension. Inspect actual processes
before inferring whether any recorded run is still active.
