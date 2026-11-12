# Prepared severity-aware auxiliary supervision

Both e43 variants fail their complete five-seed criterion and the primary
common-air contrast does not support a split-supervision benefit. These
failures motivate this separate candidate, without identifying a unique cause.
Short6080-transition smoke training completes in results/e45_smoke. Independent
final-checkpoint replay reproduces all final validation metrics over1600
cases; pipeline_check.json is explicitly diagnostic/smoke and has no matched
comparison. No full e45 policy seed or improvement result exists yet.

Five full seeds0..4 are launched in results/e45_full with five workers,
400000 requested and401280 actual transitions per seed, five intermediate
diagnostic evaluations and final-policy primary selection. Implementation
commit is c2b48aa; source hashes and exact command are in the run manifest.
The smoke was executed before the implementation commit, with bb3860a as
its Git head plus the exact uncommitted source hashes pinned in its manifest.

The hypothesis is that supervising violation magnitude can help a single-head
FCSRL representation distinguish recovery conditions. Auxiliary labels are
c+max(NH4/4-1,0)+max(maxDO/2-1,0), using next-state concentrations. Actual cost
remains c=1[NH4>4]+1[maxDO>2], cost limit12, reward and observations unchanged.
All network sizes and original encoder objectives remain inherited. This is
a different auxiliary quantity, with no binary-cost probability theorem or
novelty claim. Target scale changes together with supervision information.

The adapter uses physical float32 terminal observation fields captured before
reset and before normalization. Those concentrations are already available
in actor observations; they are not privileged extra state. Actual costs come
from the unchanged environment's original float64 threshold calculation.
Only the added auxiliary label uses float32 concentrations. Weight0 exists
only for original-label equivalence tests; the proposed candidate fixes1.

The training-only preflight in ../e42/severity_label_audit.json confirms a
shared zero-cost set and numeric support within the original symlog buckets.
That is not a policy result. severity_check.json passes original-label/RNG and
parameter equivalence, nonzero gradients, terminal/reset boundaries and
checkpoint serialization. environment_check.json compares380 actual frozen
training transitions, including reset, with exactly identical original
observations, rewards, costs, noise and resulting states. Unit checks and
smoke updates are not independent policy performance seeds. Full training
requires a successful end-to-end smoke and checkpoint replay first.

The standalone trainer scopes a runtime substitution of the training class
reference to SeverityInfoEnv, restores it on exit and never edits frozen
source files. The evaluator still uses the original quantile_env class.
The run manifest pins all e45 Python source files alongside common training
sources. Do not edit those files while any e45 runner is active.

```powershell
python Threemool/Lee/experiments/e45/check_severity.py --out Threemool/Lee/experiments/e45/severity_check.json --checkpoint results/e45_unit_checkpoint
python Threemool/Lee/experiments/e45/check_environment.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out Threemool/Lee/experiments/e45/environment_check.json
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e45/train_severity.py --out results/e45_smoke --steps 6080 --evaluations 1 --parallel 1 --jobs new:ppolag_fcsrl_severity:0
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e45/train_severity.py --out results/e45_full --parallel 5 --jobs new:ppolag_fcsrl_severity:0 new:ppolag_fcsrl_severity:1 new:ppolag_fcsrl_severity:2 new:ppolag_fcsrl_severity:3 new:ppolag_fcsrl_severity:4
```

Original encoder implementation: czp16/FCSRL commit
d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0, preserved unmodified.
See ../../code/darrow/FCSRL_LICENSE. Frozen environment, five policy seeds,
401280 transitions and the existing matched-air criterion remain the proposed
comparison. Final test41 must remain reserved.

## Queued final-policy evaluation

A separate local shell waits on the five verified training process IDs, then
runs the existing guarded matching batch below and writes comparison.json
with report.py only after all five matching jobs succeed. This is queued,
not an executed evaluation result at this snapshot. The batch rejects missing
or incomplete final checkpoints and budgets other than401280. A training or
evaluation failure must be investigated in its retained folder, not restarted
under another seed or concealed by a partial report.

```powershell
python Threemool/Lee/experiments/e44/run_pair_batch.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-roots results/e45_full/new_ppolag_fcsrl_severity --seeds 0 1 2 3 4 --out results/e45_comparison --parallel 3
python Threemool/Lee/experiments/e44/report.py --pairs results/e45_comparison/seed_0/summary.json results/e45_comparison/seed_1/summary.json results/e45_comparison/seed_2/summary.json results/e45_comparison/seed_3/summary.json results/e45_comparison/seed_4/summary.json --out Threemool/Lee/experiments/e45/comparison.json
```

Do not launch a second copy while the queued shell or matching batch is alive.
Waiting for an existing process does not count as policy training or an
improvement. Native metrics, matching traces, five seed uncertainty and all
failed results remain required before deciding on another experiment.
