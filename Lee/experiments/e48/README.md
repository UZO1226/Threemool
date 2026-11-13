# Finite-day feasibility supervision

e43/e45/e46 fail their complete matched-air extension criteria. e47's raw
PPO diagnostic is worse in cost and NH4 at matched air in all five seeds,
while reducing DO. This does not identify a sole cause, but does not support
discarding FCSRL representation learning as the solution.

The training-only audit in ../e47/feasibility_audit.json uses existing policies,
16 training dates, five members and two draws,160 cases per policy. Learned
action-conditioned scores have rank correlations.619 to.811 against the
existing discounted-max targets. Those targets include their own day-end
predictions, so this is not independent calibration evidence. Immediate costs
are concentrated in the highest score quartile, but lower quartiles are not
perfectly ordered. Scores are discounted maxima of costs0/1/2, not probabilities.

Continuation bootstraps increase mean targets by.00360 to.02584 across seeds,
with last24-step means.01316 to.09415. Maximum observed difference is.71851.
These are descriptive target effects, not proof that they harm the policy.
The hypothesis is that limiting auxiliary feasibility supervision to the
observed daily control horizon improves allocation at equal air.

Only the auxiliary target changes: use the existing max recurrence but stop
its continuation at every observed episode end, giving the discounted maximum
of costs remaining within that day. At an arbitrary rollout cutoff without
an episode end, retain the existing bootstrap. Do not edit actual terminated
or end arrays, PPO reward/cost bootstraps, advantages, reward, cost, PID limit12,
observations, networks, dynamics objective, EMA or the fixed environment.
This is a different auxiliary objective, not an assertion that continuing-task
bootstrapping is a bug in the original implementation. No extra parameter or
validation-selected coefficient is introduced.

Five policy seeds0..4,401280 transitions, final policies, the existing air-only
matching and all three quality criteria remain fixed. No final test dates or
intermediate checkpoint selection are permitted. Reused validation and weak DO
event ranking of the surrogate limit every result to exploratory simulation.
Original FCSRL d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0, remains
unmodified. Source and runtime/data provenance follow e42 and the reproduction
guide. This is personal research, not a team protocol change or novelty claim.

Before fitting, verify hand-computed daily/reset and true-terminal targets,
exact original targets when no episode end exists, unchanged PPO buffer fields,
initial parameters/RNG, inherited objective/gradients and checkpoint round trips.
Then run a6080-step smoke and independent final-checkpoint replay. Unit/smoke
results are not full policy performance seeds. Full training must wait for
those checks. Log source commit and immutable source hashes in the runner.

finite_check.json passes all specified unit checks. The6080-transition smoke
and independent1600-case final-checkpoint replay both complete successfully
in results/e48_smoke and results/e48_smoke_replay. pipeline_check.json is
explicitly smoke/diagnostic, not an independent performance seed.

Five full seeds0..4 complete in results/e48_full at implementation commit3f584a9,
five workers and401280 transitions each, with five diagnostic evaluations.
The same shell chains guarded final-policy matching in results/e48_comparison
with five workers, followed by comparison.json only after successful completion.
No full e48 improvement result exists yet. Do not duplicate the live pipeline.
The smoke ran at Git head1fd3f20 plus uncommitted sources pinned exactly in
its manifest. The full runner pins committed e48/common source files; do not
edit them while it is active.

```powershell
python Threemool/Lee/experiments/e48/check_finite.py --out Threemool/Lee/experiments/e48/finite_check.json --checkpoint results/e48_unit_checkpoint
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e48/train_finite.py --out results/e48_smoke --steps 6080 --evaluations 1 --parallel 1 --jobs new:ppolag_fcsrl_finite_day:0
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e48/train_finite.py --out results/e48_full --parallel 5 --jobs new:ppolag_fcsrl_finite_day:0 new:ppolag_fcsrl_finite_day:1 new:ppolag_fcsrl_finite_day:2 new:ppolag_fcsrl_finite_day:3 new:ppolag_fcsrl_finite_day:4
python Threemool/Lee/experiments/e44/run_pair_batch.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-roots results/e48_full/new_ppolag_fcsrl_finite_day --seeds 0 1 2 3 4 --out results/e48_comparison --parallel 5
```

All training jobs finish successfully at401280 transitions each. The five
matched-air evaluations also finish after the recovery below. All final and
diagnostic checkpoints and training logs are uploaded
to the existing draft release, with remote size/SHA256 verification in
artifacts.json. The65,810,152-byte archive SHA256 is
100d1e6e5725a9a7b93cf3fb51fb3fc984d83b2a89ccbe85fa343f637a99af72.
Original data and stochastic calibration banks are excluded. Archive presence
is not evidence that the new auxiliary objective improves the policy.

## Interrupted evaluation recovery

The initial evaluation tool handle disappears after an interrupted observation.
An actual process query finds no e48 evaluation Python jobs; some recorded PIDs
now belong to unrelated processes. The old status.json still says running and
is not evidence of liveness. No seed summary exists in that partial folder.
The interruption cause is not established. Those files/logs remain untouched.

Only evaluation restarts in results/e48_comparison_retry, with the same five
completed policies, frozen inputs, seeds, air selector, cases and criterion.
No policy is retrained and no unrelated process is stopped. The replacement
batch uses source head0882614 and chains the five-seed report on success.
interruption.json records the recovery. Before accepting a report, verify
checkpoint replay and compare retained overlapping partial evaluations with
the replacement. No outcome-based seed or checkpoint selection is allowed.

retry_verification.json passes: the job/source/freeze manifests agree and
all144105 numeric values in overlapping generated outputs match exactly.
The original files remain unchanged. All five replacement evaluations finish.
The full archive in evaluation_artifacts.json includes both completed retry
outputs and the original interrupted partial outputs, with CRC/file hashes
and remote size/SHA256 verification. There is no policy retraining.

## Complete five-seed result

All reference-air matches pass1 percent. Mean air is52027.88 versus52020.58m3.
Cost is24.266875 versus22.464125, paired difference+1.80275 with95 percent
seed-bootstrap interval[-2.82475,6.78725]. NH4 is24.97171 versus23.20868
percent, difference+1.76303 points with interval[-3.09224,7.16039]. DO is.57237
versus.43776 percent, difference+.13461 points with interval[-.16605,.44053].
All intervals cross zero, and all three quality conditions fail. Neither an
improvement nor consistent worsening is established. See comparison.json for
every seed and the native results. Reused validation, five policy seeds and
the surrogate's weak DO event ranking remain material limitations.

Changing this measured day-end target effect did not establish a better
controller here. The result does not prove that continuation is always better
or that every finite-horizon representation must fail. Original criteria and
final test reservation remain unchanged.
