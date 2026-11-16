# One-step correction: partial benefit, complete criterion fails

The complete sourcebdd9c82 pipeline finishes all five full policy seeds,
unchanged-reference checks, fixed classifier fits, held-stream audits, native
and matched-air comparisons, independent action audits and the five-seed report.
All final base checkpoint files,66 training records and five original native
evaluation records per seed match the existing FCSRL references exactly.
No extra environment training transition, test date or favorable checkpoint
is used. The original e40 freeze and e44 criterion remain unchanged.

| Matched-air mean | Original FCSRL | Fixed one-step correction |
| --- | --- | --- |
| Air, m3 per23.75h | 52020.577481 | 52021.188949 |
| Summed violation cost | 22.464125 | 22.323750 |
| NH4 violation, percent | 23.208684 | 23.141184 |
| DO violation, percent | .437763 | .357500 |

All five actual air matches pass, with absolute relative errors below.071
percent. Cost difference-.140375 has paired-policy-seed95 interval
[-.250625,-.030125]. DO difference-.080263 percentage points has interval
[-.163947,-.022368]. Both pass their predefined component conditions. NH4
difference-.067500 points has interval[-.215000,.110526], so its required
upper endpoint is not below zero. The complete improvement criterion fails.
Do not discard this condition because the total cost improved, or reinterpret
an interval crossing zero as equality. All five DO differences are negative;
four cost/NH4 differences are negative and seed3 has larger cost and NH4.

Native correction uses52153.859249m3, about.256 percent more than the reference,
and has cost22.233250, NH423.044868 percent and DO.358553 percent. Native cost
and DO also improve under the paired intervals, while NH4 remains uncertain.
Both native and matched policies are retained; matching applies its offset
after correction and selects using air only. No quality-dependent offset,
penalty, candidate radius, threshold or seed selection rescues the result.

The classifiers use316160 fitting and85120 held transitions per policy seed,
all from the same401280-transition PPO stream. Classifier seeds5500/5501 are
not extra policy seeds. Fast logits match sklearn exactly, maximum tested
probability difference is2.23e-16 and coordinate-order violation is zero.
Independent held-source and public-feature/terminal-label checks pass.
All five serialized native replays and independently reconstructed152000
action choices per seed pass. The native fraction of changed actions ranges
from.7566 to4.2454 percent. The unchanged-policy null control reproduces the
original full-day smoke. No implementation fault was identified by these
checks; their coverage does not prove the absence of every possible fault.

Strong held one-step predictive AP does not imply a larger same-air controller
gain. The correction targets the next15-minute violation and can favor an
immediate DO effect while missing slower NH4 consequences or future policy
responses. This is a follow-up hypothesis, not a proven sole cause of failure.
Training/validation season differences, stochastic-behavior versus deterministic
evaluation, action-support limits and the frozen simulator also remain possible
contributors. A longer-horizon target should be specified with a matching
one-step loss/capacity control before new fitting, rather than tuning the
successful-looking seeds or the current radius on validation.

comparison.json preserves every seed difference, sample SD and10000 paired
seed-bootstrap intervals. training_control_check.json and the runtime summaries
preserve full parity and held-data diagnostics. All source observations,
captures and stochastic banks remain outside Git and release artifacts.
Model/log/evaluation archives must include both the unchanged base checkpoint
and risk_models.pkl, not just the original policy. The original Apache license
remains required. Reused validation and multiple exploratory trials prevent
interpreting these intervals as multiplicity-adjusted independent confirmation.
The e40 DO onset/ranking limitation and lack of real-plant counterfactual
validation remain. This partial result does not complete the active goal.

The combined artifact lee-e55-one-step-correction.zip is68,760,883 bytes,
SHA25631b3d2bf541fdd8fc698cd7c77cc556f2f4f9d8509606c4b3e12ba8618a3b092.
artifacts.json verifies remote size and checksum in policy draft443333191,
asset743367629. It includes all five original/diagnostic/final checkpoints,
correction-model pickles, full logs, source manifests, held-model audits,
native/matched/all-query outputs and independent action audits. Source data,
captured arrays and stochastic banks are excluded. No e55 process remains
queued. e56 prospectively specifies a four-step cost-rate target and matching
one-step regression control using the same captured streams.
