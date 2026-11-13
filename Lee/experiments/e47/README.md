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
