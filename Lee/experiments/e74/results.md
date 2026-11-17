# e74 improves cost and NH4 but fails the unchanged DO condition

All original policy seeds0 through4 complete air matching and independent
native/matched sklearn action, full trajectory and executed-air audits.
Every152000-state wrapper audit has zero predicted component increases
relative to same-state authenticated matched e63. Reference-only and frozen
offset NN null controls reproduce their declared complete trajectories.

| Matched metric | e63 mean | e74 mean | Difference95% paired-seed interval |
| --- | --- | --- | --- |
| Air, m3 per23.75h | 52018.537779 | 52014.183291 | [-35.993161,27.038866] |
| Violation cost | 21.060125 | 18.228125 | [-5.343250,-.908875] |
| NH4 violation percent | 21.844737 | 18.895658 | [-5.592895,-.942895] pp |
| DO violation percent | .323816 | .291842 | [-.081710526,.000526316] pp |

All air matches pass and every seed improves cost and NH4. Mean DO falls,
but seed3 increases by.004605263 points and seed4 is unchanged. The DO upper
endpoint is positive. The joint criterion therefore fails; do not round it
to zero, relax nonincrease or pool more favorable blocks. No exposed5 through9
evaluation is triggered, as preregistered. The active goal remains unfinished.

Native means are53761.497094m3, cost16.198375, NH4 16.734868% and DO .316053%.
Native air remains above matched e63; its quality does not replace the actual
air comparison. initial_verification.json reconstructs all recorded episode
metrics, actual validation dates, air-only choices and10000 paired policy-seed
resamples with RNG2026. The same original401280-step NNs, obs11, horizon4
heads and frozen e42 world are preserved.

The declared final-action diagnostic again finds seed3's post-offset DO
prediction increasing on19265 visited states, mean+.000646142 relative to
same-state matched e63. The matched-reference guard is pre-offset and cannot
constrain the additional matching offset or guarantee recursive outcomes.
This is descriptive model evidence, not a causal decomposition of actual
DO violations. Native uncertainty also remains, so offset effects alone
are not established as the full failure mechanism.

## Secondary e73 mechanism contrast

mechanism_contrast.json and analysis/contrast_matched_anchor.py recompute
the preregistered secondary comparison from both complete audited reports.
Both matched policies target the same authenticated e63 air. e74 minus e73
matched cost is-1.921750 with interval[-4.391875,-.414875], NH4 is-2.014474
points with[-4.616974,-.438026], and DO is-.008421 points with
[-.026053,.005789]. The secondary DO interval includes zero and does not
rescue the primary failure. The anchor change includes its coherent geometry
and calibration implications, not an isolated causal ablation.

Native versus native air decreases2866.807491m3 and DO decreases.044079
points, but native NH4 increases.374868 points. These native tradeoffs remain
separate from matched-policy gains. No new environment query generates this
secondary report. Its small report and generator are in Git; they supplement
the immutable primary model archive.

## Delivery and limits

Full primary weights, configurations, queries, logs and all audits are in
private draft443333191, asset757838592, lee-e74-matched-anchor.zip,
14218576 bytes, SHA256
e3eed852e5168f1c3fb25281e90185842015bd5739ec84464bc07dcd917f65a0.
artifacts.json binds individual file hashes and verified remote size/digest.
No observations, training captures, residual banks or PIT banks are uploaded.
The runtime manifest pins exact sources, code commit1f99cdd.

All previous failures remain preserved. Risk-rate plateaus, action support,
rare DO events, occupancy changes and recursive model bias remain unresolved.
Reused validation40 is exploratory, and final test41 is unused. An auxiliary
model trained on simulator captures would be an agent-side predictor, not
evidence that the original DARROW environment's safety forecasts are repaired.
