# Calibration-only support limitation

support_audit.json is an additional diagnostic on the30 inner-calibration dates
only. No validation/test outcome or new fit enters it. It does not select
e51 settings, change its adoption gate or bound free-running forecasts.

For each real current state, the frozen quantile sampler cannot exceed its
lowest/highest predicted quantile after original blending/clipping. For NH4,
an event is impossible if the upper support is at most4. For DO, it is
impossible if every upper zone support is at most2. Conversely an event is
forced if the corresponding lower support already violates. Average these
indicators across the five equally weighted members to get an optimistic
attainable event-probability interval. Squared distance of the binary target
from that interval lower-bounds the empirical Brier for this observed-state
task, even allowing arbitrary row-specific noise. A shared empirical PIT bank
can be more restricted. This is not an independent population risk estimate.

There are157 observed NH4 violations, including9 new onsets, and306 DO
violations, including26 new onsets, across2850 inner transitions. At alpha.5,
1/9 NH4 and9/26 DO new onsets have no supporting member. At alpha.75,
2/9 NH4 and19/26 DO new onsets are unsupported. Reweighting PIT values inside
the existing support cannot generate those observed-state events. It can still
alter other events and recursive states, so this is not proof that e51 or any
full-day model must fail its aggregate gate.

At alpha0,220/306 DO violation steps are unsupported, versus11/306 at alpha.5.
Larger alpha retains ongoing measured DO violations but can narrow support
for new events. Observed-state persistence and autonomous prediction remain
different tasks. Do not equate aggregate frequency recovery with onset skill.

All individual member/state rows remain local. The compact report gives
support coverage, concentration gaps, optimistic probability bounds and event
counts for every original alpha. Five members are not five independent
policies or2850*5 independent dates. No threshold is tuned on these counts.

```powershell
python Threemool/Lee/analysis/audit_quantile_support.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out Threemool/Lee/experiments/e51/support_audit.json
```
