# e68: Fresh policy-seed baseline preparation

Prospective replication block, 2026-11-17. Establish seeds5,6,7,8,9 before
examining any new candidate's quality on those policies. This is baseline
preparation, not a new extension, a successful replication or replacement
of the original seeds0 through4. e67's original five-seed comparison is still
running. Its result cannot be rescued by ignoring an unfavorable seed block.

Keep the e42 environment/data/inference freeze and original FCSRL training
configuration:401280 actual environment steps, eight environments,760-step
updates, cost limit12, final checkpoint only and five scheduled validation
evaluations. Final test41 is excluded. Train each seed twice: unchanged
original trainer and unchanged passive-capture trainer. Require exact final
checkpoint files, configuration, seed, budget and stored native evaluation
parity before fitting its horizon4 risk regressors. This verifies that the
capture mechanism does not change the policy for the fresh seeds either.

Use the existing training and capture implementations without modifying
their historical sources. A new runner permits exactly seeds5 through9;
the historical e42 runner accepts only0 through4. This is an additional
within-Lee research block, not a change to the shared evaluation protocol.
Retain actual seed identities rather than silently relabeling them0 through4.
Run at most two policy trainers concurrently and preserve progress/logs.

The primary replication comparator must be freshly reconstructed e63:
each new NN plus its own training-capture-fitted horizon4 rate models,
original five-candidate radius0.25 zero-penalty controller and the unchanged
air-only offset matching to that seed's native original-FCSRL air. Then
match a predeclared candidate to the achieved e63 air, including the fixed
e63 offset in the reference. Do not use original FCSRL as the sole new
primary baseline. All native results and failures remain reported.

Any later candidate replication requires all five actual-air matches, cost
and NH4 paired policy-seed95% interval upper endpoints below zero and DO at
most zero,10000 paired resamples with RNG2026. Preserve separate original
and fresh blocks; pooled reporting is secondary and cannot override a failed
required block. No checkpoint, seed, threshold, model, radius or risk-weight
selection by fresh quality is allowed. Reused validation40 remains exploratory
even with fresh policy random seeds, not a new observational holdout.

Baseline training is useful for later predeclared candidates even if e67
fails. Do not call fresh-seed preparation an improvement, a complete goal,
or evidence that environment DO onset prediction is repaired. Source data
and capture arrays stay local; required checkpoints/logs eventually go to
the already authorized private draft storage with verified checksums.

## Reproduction and current execution

From the workspace, with the threemool Python environment and numerical
library threads fixed to1, launch into unused output paths:

```powershell
python Threemool/Lee/experiments/e68/run_baselines.py --workspace C:/Users/user/Downloads/IoT_project2 --out results/e68_fresh_baselines
python Threemool/Lee/experiments/e68/prepare_rates.py --workspace C:/Users/user/Downloads/IoT_project2 --baselines results/e68_fresh_baselines --out results/e68_fresh_rates
```

The second command requires the first command's completed summary and exact
parity records; it is queued behind the live training runner, not running
on partial checkpoints. It first reproduces the original e55 one-step
capture/provenance fitting, then fits unchanged e56 horizon4 regressors from
updates1 through52 and diagnoses held updates53 through66. The one-step
models are not used or selected as fresh h4 controllers. No added environment
training steps belong to either offline fit. This preserves the earlier
pipeline rather than weakening capture guards to avoid provenance fitting.

Current output is not a completed artifact or replicated policy gain.
study_status.json is a progress snapshot; inspect actual process liveness
before restarting. All historical trainer/model sources remain unchanged.
