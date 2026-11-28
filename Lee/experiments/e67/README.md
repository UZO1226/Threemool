# e67: A larger fixed risk-only action radius

Prospective hypothesis, 2026-11-17. e63 proves that removing its action-change
penalty can improve matched-air quality. e65's five completed comparison
calculations show only a small effect from refining the grid within the same
radius; its independent audits are still pending and it is not a verified
improvement. Test whether the radius, rather than interior grid resolution,
limits risk-guided allocation. This is a new experiment, not an adjustment
of e65 or a new success rule.

Keep the five401280-step e63 base policies and their own unchanged horizon4
NH4/DO regressors, observation11 inputs and zero penalty. Use exactly nine
offsets in order0,-.125,.125,-.25,.25,-.375,.375,-.5,.5, clipped to[-1,1].
The original e63 candidates remain first, preserving its action on equal
predicted risk. Radius0.5 is fixed before held-state or validation outcomes.
Do not search radii, weights, thresholds, horizons or alternative tie rules.

First authenticate all85120 held training states for every seed, verify the
larger-radius bound and exact independent sklearn actions on a fixed1024-row
subset. Report how many actions and predicted risks change relative to e63.
No counterfactual labels are available, so predicted gains are not evidence
of improved quality or causal support. Wider changes may be less supported
by logged policy actions, an uncertainty to retain rather than hide.

If all held actions are identical across all five seeds, record identity and
defer an expensive comparison. Otherwise run seed0 native smoke, full
independent action/trajectory and flat-risk null checks, then all five
native and matched-air comparisons plus independent audits. The primary
reference is e64-authenticated matched e63 with each fixed offset. Keep the
quality-blind e44 air matcher and target each e63 per-seed actual air volume.
Retain original FCSRL secondary metrics and all unfavorable seed outcomes.

The unchanged improvement rule requires all air matches and10000 paired
policy-seed bootstrap samples, RNG2026, with cost and NH4 interval upper
endpoints below zero and DO upper endpoint at most zero versus e63. Native
air/quality must be reported. No new model fitting or environment training
is added to the401280-step policy budgets. Reused validation40 is exploratory
and test41 is excluded. Feasible fresh policy seeds remain required after
any initial success. e42 stays fixed; e66 is rejected, not silently substituted.
