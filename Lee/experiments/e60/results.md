# Feedback horizon and temporal-noise diagnosis

All16 fixed calibration conditions complete under2e4bd6d. Both reset1/e59
and reset95/selected-inner identity controls pass, as does exact tuple
permutation/inverse. No fitting, validation evaluation or final test occurs.
summary.json preserves full scores, event/onset metrics and every paired
date-bootstrap contrast. No e60 job remains live.

| Model | Reset interval | NH4 Brier, original | DO Brier, original | DO Brier, shuffled |
| --- | ---: | ---: | ---: | ---: |
| e51 | 1 | .004778 | .092971 | .093186 |
| e51 | 4 | .010763 | .100254 | .100322 |
| e51 | 16 | .021652 | .103490 | .103872 |
| e51 | 95 | .080070 | .104576 | .105259 |
| e58 | 1 | .004778 | .074999 | .075973 |
| e58 | 4 | .010761 | .081581 | .084721 |
| e58 | 16 | .021660 | .086813 | .090434 |
| e58 | 95 | .072797 | .092882 | .095343 |

More frequent factual state resets reduce descriptive error, especially NH4.
Those resets expose intermediate observations and cannot be deployed as an
improved free-running world. This is evidence of a feedback-associated gap,
not identification of its sole cause or a physical causal response.

Every shuffled-minus-original NH4/DO Brier95-percent circular five-day
bootstrap interval includes zero. For DO at reset95, e51's difference is
+.000683 with interval[-.001086,.002342]; e58's is+.002460 with interval
[-.003136,.011312]. Randomizing within-day order is not a demonstrated repair.
The permutation preserves each complete noise tuple and daily multiset, so
the experiment does not test conditioning the distribution on episode context.

Calibration dates also estimate PIT and are not independent evaluation.
The e59 context/prevalence shift and e58 recursive failure remain unresolved.
Prior e17/e20 contraction families, e31 joint horizons, e35/e37 assimilation
and e50 forecast-state augmentation already failed their adoption checks.
Do not relabel this diagnostic as another model improvement or repeat those
changes without a new concrete hypothesis.

A next candidate can test a context-conditioned noise distribution, rather
than time permutation alone: select whole calibration-day paths using only
known episode-start state/exogenous context and fitting-only scales. Fix its
features, neighbor count, leave-one-day-out calibration controls, primary
parent and complete adoption criteria before outcomes. The current e42 world
and all policy results remain unchanged; the overall goal remains active.
