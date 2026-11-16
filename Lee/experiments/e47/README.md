# Equal-air PPO diagnostic before another extension

e43, e45 and e46 do not establish a complete matched-air improvement over
the fixed FCSRL reference. e46 reduces DO with a paired interval below zero,
but its total cost/NH4 conditions fail. Before training another candidate,
test whether the already-trained raw-observation PPO policies perform better
at the same reference air. This diagnoses a missing ablation; it is not a
new FCSRL extension or an opportunity to relabel PPO as an extension.

Use the five completed frozen-environment PPO final policies from e42,
seeds0..4 and401280 transitions each, with their original FCSRL seed references.
No model or policy is fitted here. Reward, actual cost, limit12, frozen
environment, evaluation cases and quality-blind air matching are unchanged.
The kind=diagnostic path permits different representations while checking
identical PPO settings, seed, budget, sources, checkpoint and freeze identity.
Native replay must reproduce final saved metrics. No intermediate checkpoint
selection, test dates or revised pass criterion is allowed.

Report exactly five paired policy seeds with the existing10000-resample seed
bootstrap and native/adjusted results. A successful diagnostic could motivate
a controlled representation ablation, but would not prove its cause or finish
the user's extension goal. A failed diagnostic also remains evidence rather
than an external blocker. Validation reuse and weak DO event prediction of
the frozen surrogate limit all claims to exploratory simulation.

```powershell
python Threemool/Lee/experiments/e44/run_pair_batch.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-roots results/e42_baselines/new_ppolag --seeds 0 1 2 3 4 --out results/e47_ppo_diagnostic --parallel 5 --kind diagnostic
python Threemool/Lee/experiments/e44/report.py --pairs results/e47_ppo_diagnostic/seed_0/summary.json results/e47_ppo_diagnostic/seed_1/summary.json results/e47_ppo_diagnostic/seed_2/summary.json results/e47_ppo_diagnostic/seed_3/summary.json results/e47_ppo_diagnostic/seed_4/summary.json --out Threemool/Lee/experiments/e47/comparison.json
```

Original policy artifacts are in ../e42/new_ppo_artifacts.json and
new_fcsrl_artifacts.json. Source/data/runtime/freeze provenance follows e42
and ../../analysis/DARROW_REPRODUCTION.md. Air is m3 over23.75h, not measured
electricity. This is personal diagnostic work, not a team protocol change.

The five-worker batch is launched in results/e47_ppo_diagnostic at source
commit34b9f26. Its shell writes comparison.json only after all five pairs
finish successfully. No new policy training occurs. Do not duplicate the
running batch or infer completion from an old status file. PPO architecture
and initialization differ from representation-augmented FCSRL, so this does
not isolate the encoder objective as the sole possible cause of a difference.

## Complete diagnostic result

All five pairs finish and match air within1 percent. Native PPO air is
76087.49 versus FCSRL52020.58m3. At matched air52013.01 versus52020.58m3,
PPO cost is32.399375 versus22.464125, paired difference+9.93525 with95 percent
seed-bootstrap interval[4.779375,14.518875]. NH4 is34.09303 versus23.20868
percent, difference+10.88434 points with interval[5.16092,15.88987]. All
five cost and NH4 differences are positive. DO is.01158 versus.43776 percent,
difference-.42618 points with interval[-.94197,-.06553]. The full criterion
fails, and the trade-off must remain visible rather than claiming dominance
on both constraints.

This does not support replacing FCSRL with raw PPO as the solution. It also
does not prove representation learning is the sole cause, because architecture
and initialization differ. No extension improvement or real-plant safety is
established. comparison.json retains every seed and native/adjusted metrics.
All generated evaluation queries are remotely verified in
evaluation_artifacts.json. No new weights or original data are uploaded.

Before selecting another actor intervention, audit existing learned feasibility
scores on fixed training-day rollouts, including their relationship to observed
discounted-max targets and the contribution of day-end continuation bootstraps.
This diagnostic does not fit a policy, change the fixed environment, modify
targets in an existing run, or use final test dates. A finite-day alternative
is a different objective, not automatically a bug fix to the existing code.

That training-only audit now completes. feasibility_audit.json preserves all
five policies' score quartiles, rank correlations and day-end effects, with
unchanged checkpoints and no fitting. Scores rank high-risk transitions, but
the continuation targets partly depend on the same learned head. Their rank
correlations are therefore not independent calibration evidence. Removing
day-end continuation would change mean labels by.00360 to.02584 and last24-step
means by.01316 to.09415. These measured target effects motivate e48's separate
finite-day auxiliary hypothesis, without proving its policy benefit.
