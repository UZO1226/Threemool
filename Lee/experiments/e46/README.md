# Colder-day exposure with unchanged FCSRL

This separate sampling hypothesis follows e43 and e45, which fail their
complete five-seed matched-air criteria. e46's complete comparison also fails
the full criterion, while its DO-only condition passes. See the results below.
The existing risk-day sampler failed in e15 on a different model. We do not
erase that failure, repeat it under another name or claim a new RL algorithm.

Observed training temperature averages19.714C versus15.962C in reused
validation. Initial NH4 violations occur in7.45 versus30 percent of days.
Baseline final training rollout costs are below validation costs, but these
changing-policy summaries do not identify a causal failure mechanism.
See ../e45/training_shift_audit.json. Validation is diagnostic, not independent
confirmation, and final test41 is unused.

The hypothesis is that more exposure to colder observed training days can
improve generalization of the original FCSRL policy at matched air. Rank only
training days by mean recorded temperature and give the coldest quarter half
the reset probability, and all remaining days the other half. Stable ties use
original date order. The quarter/half allocation follows the pre-existing
e15 exposure strength; no cutoff or probability is searched on validation.
All188 training days retain positive probability. Weighted observed mean
temperature is18.801C, still above validation, so this does not remove the
distribution gap or assert coverage of unseen conditions.

Networks, feasibility targets, reward, actual cost, limit12, frozen model,
401280 transitions, five policy seeds and final-policy evaluation remain
unchanged. Only reset-day sampling changes. The PID therefore receives a
different training-day cost distribution; this is part of the intervention,
not a guarantee on uniform training or validation safety. Uniform validation
evaluation uses the original frozen environment and the existing air-only
matching algorithm and three-quality-condition criterion. Report native and
adjusted policies, both constraints and all five paired seed intervals.

Before any fit, check stable ties, finite/positive normalized weights, no
validation/test sampling, actual reset frequencies and exact outputs through
a full95-transition day at common fixed cases. Then run a6080-step smoke,
load its final checkpoint and independently replay validation. Full training
must wait for those checks. No original FCSRL or frozen source is modified;
the standalone trainer only scopes its training class reference substitution.
Sampling metadata is written to sampling.json alongside every run config.

Original FCSRL d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0, is
preserved. This remains exploratory personal work, not a team protocol change.

## Checks and full execution

sampling_check.json passes fixed-case380-transition output equivalence,
training-only support, stable ties, nonfinite input rejection and1024 actual
reset draws. The observed cold fraction is.52930 versus the specified.5;
this is a sampling check, not a performance seed. The6080-transition smoke
finishes in results/e46_smoke, and an independent1600-case final-checkpoint
replay reproduces all recorded validation metrics. pipeline_check.json is
explicitly smoke/diagnostic and has no matched performance comparison.

Five full seeds0..4 launch in results/e46_full with source commit98ad497,
five workers,400000 requested/401280 actual transitions,66 updates and five
diagnostic evaluations. Only final policies determine the primary comparison.
The smoke ran at Git head b5e72d1 plus uncommitted sources pinned exactly by
its manifest. Full source files are committed and hashed by the full runner.
Do not edit e46/common training Python sources while that runner is active.

The same shell chains the guarded final-policy matching batch and report
after successful training. Failure stops the chain and preserves logs. Do
not launch another copy because a status file is old or observation times out.

```powershell
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e46/train_cold.py --out results/e46_full --parallel 5 --jobs new:ppolag_fcsrl_cold:0 new:ppolag_fcsrl_cold:1 new:ppolag_fcsrl_cold:2 new:ppolag_fcsrl_cold:3 new:ppolag_fcsrl_cold:4
python Threemool/Lee/experiments/e44/run_pair_batch.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-roots results/e46_full/new_ppolag_fcsrl_cold --seeds 0 1 2 3 4 --out results/e46_comparison --parallel 3
python Threemool/Lee/experiments/e44/report.py --pairs results/e46_comparison/seed_0/summary.json results/e46_comparison/seed_1/summary.json results/e46_comparison/seed_2/summary.json results/e46_comparison/seed_3/summary.json results/e46_comparison/seed_4/summary.json --out Threemool/Lee/experiments/e46/comparison.json
```

Use fresh output paths to reproduce. Runtime and prepared-input hashes,
188/40/41 chronological splits, preprocessing and e40 freeze reconstruction
are in [the reproduction guide](../../analysis/DARROW_REPRODUCTION.md) and
[e42](../e42/README.md). Recorded runtime is Python3.11.15, sklearn1.9.1,
Torch2.15.0 CPU with one numerical-library thread per worker. Air is volume
in m3 over23.75h, not measured electricity. Uniform evaluation uses1600
member/date/noise cases per policy, not1600 independent policy seeds.

All five full training jobs and the guarded evaluation batch complete
successfully in results/e46_full and results/e46_comparison. Final
checkpoints, diagnostic checkpoints, sampling settings and full training logs
are uploaded to the existing policy draft release and verified remotely in
artifacts.json. The65,799,004-byte archive SHA256 is
13a93a5c68fd3726343f1b4cc5cdeafdca1bf8481d52ccf5f595849dc2053617.
No original data or calibration banks are included. Archive existence does
not imply improvement and the reserved final test remains unused.

## Complete matched-air result

All five comparisons now finish successfully. Every reference-air match is
within1 percent. Mean air is52014.09 versus52020.58m3. Cost is24.65 versus
22.464125, difference+2.185875 with95 percent paired seed-bootstrap interval
[-2.295875,6.667625]. NH4 is25.64921 versus23.20868 percent, difference
+2.44053 points with interval[-2.20618,7.08724]. DO is.29816 versus.43776
percent, difference-.13961 points with interval[-.3025,-.01921]. Thus DO's
condition passes, but total cost and NH4 do not. The full criterion fails.
Do not describe the DO-only benefit as a complete improvement.

Native mean air increases65136.17 versus52020.58m3, and its paired air
interval is strictly positive. Lower native cost does not establish a better
equal-air operating curve. The temperature exposure intervention changes the
training/PID distribution and does not identify the sole cause of any effect.
All seed directions, native results and paired intervals are retained in
comparison.json. Five seeds and repeatedly reused validation do not establish
real-plant safety or a population confidence guarantee.

Full generated air queries and evaluation cases are remotely verified in
evaluation_artifacts.json. Final test41 remains unused. e47 next evaluates
the existing PPO checkpoints at the same FCSRL air as a diagnostic ablation,
without training or calling PPO a new FCSRL extension.
