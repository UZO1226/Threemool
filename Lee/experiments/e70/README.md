# e70: Diagnose transport of the air constraint

Prospective failure-analysis question, 2026-11-17. e69 matches mean action
on logged fitting states, but its native actual air differs from native e63
by about3046.989m3 per episode on average. Matched-air calculations fail
the joint criterion, pending their final independent audits. Do not retune
e69 on this result. Run this diagnostic only after the complete independent
study confirms failure, not after a partial seed or smoke outcome.

No new controller, model fitting, threshold selection or training is planned.
Replay both unchanged native controllers for every original policy seed on
the same frozen e42 environment and validation noise paths. On each visited
state, observe both policies' deterministic actions without using the observer
to control that rollout. Require replayed complete episode summaries to
reproduce the already stored e63/e69 native results. Actual air accumulation
must independently equal the mean action conversion.

Let X0 and X1 denote states visited by e63 and e69 respectively. Decompose
actual native mean-action difference by both exact algebraic identities:

```text
E_X1[a1] - E_X0[a0]
  = E_X0[a1-a0] + E_X1[a1] - E_X0[a1]
  = E_X1[a1-a0] + E_X1[a0] - E_X0[a0]
```

Report the symmetric average of the two mapping terms and the two occupancy
terms, whose sum equals the actual difference. This is a model/trajectory
decomposition, not causal identification in a physical plant. Record four
fixed current-state regimes using existing research thresholds only:
NH4>4 and maxDO>2, giving neither, DO only, NH4 only and both. For both
occupancies report each regime's count and both conditional action means.
Do not choose regimes or thresholds from outcomes, resample individual
transitions as independent evidence, or treat logged static equivalent
volumes as executed volumes.

Use all five policy training seeds0~4. Report10000 paired-seed bootstrap
intervals, RNG2026, for the air components, clearly separate from the original
quality criterion. Each seed requires two152000-transition evaluation replays,
304000 evaluation transitions and zero additional environment-training steps.
Save aggregates and source/model/controller digests only, no observation
arrays. No candidate is selected by this script. The diagnostic may inform a
separately preregistered hypothesis; it cannot rescue e69 or lower its gate.

Final test41 is excluded. Validation40 is reused exploratory data. The fresh
policy block in e68 remains separate and has no candidate-quality result.
