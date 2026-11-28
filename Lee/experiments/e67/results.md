# Larger-radius result

All five complete comparisons and independent native/matched sklearn
action/trajectory audits pass their implementation checks. The primary
joint improvement criterion fails. Do not adopt this controller over e63.

| Matched-air measure | e63 | Radius0.5 candidate | Paired difference95% interval |
| --- | --- | --- | --- |
| Air, m3 per23.75hours | 52018.537779 | 51996.852575 | [-44.087739,5.240331] |
| Violation cost | 21.060125 | 21.730500 | [-.506625,1.910125] |
| NH4 violation, percent | 21.844737 | 22.741316 | [-.427105,2.416316] |
| DO violation, percent | .323816 | .132895 | [-.448289,-.018289] |

All five actual air matches satisfy the unchanged tolerance. All five DO
differences favor the candidate, but cost and NH4 worsen in seeds0,3,4 and
improve in seeds1,2. Do not discard any seed or reinterpret the DO-only gain
as the requested joint improvement. Intervals retain10000 paired policy-seed
bootstrap resamples with RNG2026. This is exploratory reused validation.

Native air rises to60138.226996, with cost15.313625, NH4 15.840526% and
DO .279079%. Relative to e63's native air55039.345248, the larger radius
substantially changes the air operating point before offset matching.
Native cost reduction therefore cannot establish matched-air efficiency.
At matched air, NH4 difference+.896579 points and cost+.670375 fail the
declared upper-interval rules even though DO difference-.190921 points
passes its nonincrease rule. The result suggests studying allocation under
an air constraint rather than treating larger native corrections as a gain;
it does not prove the cause of the failure or validate a new method.

Held85120-row actions changed16124,22591,18853,43277,23447 relative to e63,
with independent fixed sklearn subsets exact. The native smoke reproduces
the authenticated e63 baseline, independently reloaded new controller and
full flat-risk null. Full audits independently check152000 actions per mode
and seed against original sklearn regressors, exact95-step trajectories,
input preservation, radius and executed-air accounting. All pass.

comparison.json, per-seed audits and artifacts.json preserve the complete
failure. The verified existing private policy draft asset includes unchanged
base/rate weights, native/matched trajectories, full air queries, logs,
configs and provenance. It excludes source observations, training captures
and PIT banks. Partial-study archival refusal and completed archive digest
verification both pass. No new weights were fitted for this experiment,
all original policies trained401280 steps, and final test41 remains unused.
