# Final DARROW research objective audit

The verified stopping condition is the e63 improvement in the predefined
DARROW research simulation comparison. It is not termination because trials
failed or because a model is imperfect. All requested investigation, model
checks, fixed-world multi-policy-seed comparison and research deliverables
are accounted for below. This does not certify a real wastewater plant.

| Requirement | Current evidence | Assessment |
| --- | --- | --- |
| Review existing work and repository rules | AGENTS.md, CLAUDE.md, RULES/BRANCH_RULES.md; checked Lee-only branch scope | Verified |
| Investigate DARROW process models and compatible public implementations | DARROW_SOURCE_REVIEW_20261112.md and DARROW_PUBLIC_MODEL_SEARCH_20261112.md, including original source limitations and BSM1/BSM2 compatibility distinctions | Investigation completed; no claim that compatible public code cannot exist |
| Improve and validate the empirical environment before policy comparison | e40/summary.json, e40/audit.json and e42/environment_freeze.json; original e16 adoption gate preserved | All six original conditions pass, four of five model members calibrated |
| Examine NH4 underprediction separately | Earlier selected e16 predicts8.505263 percent versus17.315789 observed; e40 predicts13.499342 percent, reducing absolute frequency error8.810526 to3.816447 points | Improved frequency calibration; residual underprediction remains |
| Check error accumulation and air response | e40 independent full-day reconstruction and common-noise air-order checks; e49/e59 observed-state diagnosis and e60 reset-horizon controls | Checked; recursive error remains a documented limitation |
| Compare simple forecasts and separate event probability/timing/uncertainty | e40/audit.json and event_reference_audit.json, plus later fixed diagnostics | All quantities reported, including unfavorable DO versus persistence and missed onsets |
| Distinguish implementation defects from model failure | e40/bounds_check.json reproduces the quantile monotonicity defect and corrected zero reversals; e58/e61 independent reconstructions preserve their failures | Verified separation, no silent library or historical source edits |
| Freeze the environment and isolate agent effects | e42 manifest authenticates models/bank/data/inference sources; all12 completed agent/control reports share its canonical identity | No environment change explains e63's agent contrast |
| Hypothesis, implementation, learning and validation | e55/e56 trained own-policy risk components; e63 specification6b8df22 precedes implementation/new trajectories and uses their authenticated fixed models | Learned extension tested without pretending reused weights are new PPO training |
| Compare multiple policy seeds at the same air | e63/comparison.json, five original401280-step policy seeds, max actual air mismatch.085150 percent | All four original simulation criteria pass |
| Independently verify native and matched behavior | e63 study action audits, five seeds each152000 native and152000 matched sklearn-reconstructed decisions, trajectories and executed-air accounting | All pass; config/model/checkpoint/source/budget guards retained |
| Preserve native tradeoffs and uncertainty | e63/results.md, comparison.json and penalty_contrast.json | Native air+5.803 percent explicitly remains an air/quality tradeoff |
| Preserve failed results and model/policy seed distinction | All12 completed agent/control reports and e16 onward model families, DARROW_AGENT_TRIALS.md and figure | Eleven prior agent/control failures retained; no member/draw/date pooling as policy seeds |
| Preserve final test | Official188/40/41 split; fit158/calibration30, validation40 only; train/validation-only environment constructors and captured-day guards | Test41 excluded from fitting, selection, evaluation and confirmation |
| Push permitted code/results and preserve original data | lee/darrow-safety-calibration, Lee-only changes, scope and paper checks; no newly tracked data/model array files | Verified; main, team protocol, other members and Paper remain unchanged |
| Record reproducible artifacts and locations | e63/artifacts.json, DARROW_REPRODUCTION.md, reconstruction helpers and exact archive hashes | Complete archive verified locally and remotely; raw data/captures/banks excluded |
| Retain checksum protection when sources move | replay_trust_portable.py reuses the tested source-location adapter; seed0 complete native/matched replay and108 source verifications | Integration replay passes in this Windows checkout; not an all-platform portability claim |

## Environment evidence and remaining scientific limits

The model adopted before policy comparisons passes the original prospective
e16 gate, not a retroactively weakened criterion. Relative to the original
GBM, recursive normalized MAE improves1.026065 to.775476, NH4 MAE2.482256 to
1.977722, NH4 frequency absolute error11.168421 to3.816447 percentage points,
and DO frequency error.642105 to.436842 points. Persistence normalized MAE is
.963913. Compared with the earlier underpredicting e16 candidate, the NH4
frequency improves but normalized MAE is worse than its.703608; this is not
uniform dominance over every earlier model or every individual output.

The e40 event audit reports NH4/DO Brier.126486/.013327 versus original
.217684/.021074. Both paired-date differences favor e40 over the original
model. DO Brier is nevertheless worse than persistence.011579, with difference
interval[.000046,.004223], and no new DO onset is matched within four steps.
These unfavorable findings are part of the requested separate assessment.
They are not claimed solved, and later e58/e61 models fail their stronger
predeclared criteria and are never adopted. Environment assessment and agent
improvement are kept distinct, rather than equating an agent win with accurate
plant physics or every possible safety prediction metric improving.

## Agent evidence meeting the requested stopping condition

The original air/cost/NH4/DO criterion is unchanged. With the same frozen
world and original five final FCSRL policies, e63 matched means are
52018.537779m3 air,21.060125 cost,21.844737 percent NH4 and.323816 percent DO.
Original means are52020.577481m3,22.464125,23.208684 percent and.437763 percent.
All five seeds improve each quality measure. Paired95-percent intervals are
cost[-2.204125,-.713375], NH4[-2.231184,-.549342] points and DO
[-.200526,-.052105] points. Same-base/rate-model penalty1 secondary comparisons
also favor e63, but do not substitute for that primary result.

The improvement belongs to the budget-matched policy, whose constant offset
is applied after the wrapper and changes previous-air feedback. Native air
instead rises5.803 percent. Costs remain above the12-step limit on average;
improvement is not a claim of zero violations or a safe controller. There is
no new independent date test, fresh policy-seed confirmation, electricity
measurement, legal discharge compliance assessment or actual-plant deployment.
Repeated validation and many investigated hypotheses create selection
uncertainty beyond the conditional five-seed intervals.

## Delivery and verification

The complete e63 archive is14,263,028 bytes, SHA256
183c56ee7c864a04adbd337929040d557c37627680cbdb030a198c4443963aa3,
policy draft443333191 asset753857743. All182 indexed file hashes and the ZIP
integrity check pass. The remote asset's size/digest match, and draft status
is retained. The archive excludes source observations, captured arrays and
stochastic banks. Previous failed artifacts and exact PIT reconstruction
methods remain available in their indexes and DARROW_REPRODUCTION.md.

The research stopping condition is supported by the completed original model
gate, explicit unfavorable diagnostics, fixed-world verified agent improvement
and preserved deliverables. Optional future DO forecasting or independent
team-test confirmation remains future research; it is not falsely reported
as completed and no forbidden test is used to obtain a stronger claim.
The goal status can be completed only after final branch publication is
confirmed. No experiment or artifact upload is left running.
