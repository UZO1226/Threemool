# e73 fails the unchanged DO condition

All five original policy seeds complete actual-air matching and independent
native/matched action, trajectory and executed-air audits. Every executed
trajectory's pre-offset wrapper satisfies the predicted NH4/DO component
guard, with zero increases on all152000 states per seed per mode. These
implementation checks do not establish the joint improvement criterion.

| Matched metric | e63 mean | e73 mean | Difference95% paired-seed interval |
| --- | --- | --- | --- |
| Actual air, m3 per23.75h | 52018.537779 | 52013.693873 | [-31.359131,17.253337] |
| Violation cost | 21.060125 | 20.149875 | [-1.502750,-.317750] |
| NH4 violation percent | 21.844737 | 20.910132 | [-1.554474,-.314737] pp |
| DO violation percent | .323816 | .300263 | [-.055658,.004737] pp |

All air matches pass and cost/NH4 improve for each seed. Mean DO falls, but
seed3 increases by.019078947 points. The DO interval upper endpoint exceeds
zero, so the prospectively fixed joint criterion fails. Do not substitute
favorable means, round the upper endpoint to zero, pool other seeds or change
the nonincrease rule. As preregistered, no exposed-seed follow-up is triggered
by this failed original block. The goal remains active.

Native e73 uses56628.304585m3, cost15.884125, NH4 16.360000% and DO .360132%.
Its higher air use and lower native cost do not substitute for the matched
comparison. initial_verification.json independently reconstructs all recorded
episode aggregates, validation dates, air-only matching decisions and10000
paired policy-seed resamples with RNG2026. All five policy NNs still use401280
steps, obs11, unchanged horizon4 regressors and the same frozen e42 world.

The declared post-offset diagnostic finds seed3's final action increases
predicted DO risk on14053 visited states relative to same-state authenticated
matched e63. Its mean predicted DO change is+.000641224. The wrapper guard
is relative to e63 native and applies before the additional air offset;
it does not constrain the final matched action. These same-state predicted
changes are descriptive, not an exact causal decomposition of water quality.
Native DO also remains unfavorable relative to the matched reference, so
the offset alone is not established as the failure's cause. Recursive state
shift and risk-head misspecification remain possible contributors.

The completed archive is in private draft443333191, asset757737999,
lee-e73-component-guard.zip,14425505 bytes,
SHA256 289550c40787fe5feebf22445999e833c95bbdba6a943194161feb4c301ff813.
artifacts.json binds all model weights, fit parameters, full queries, logs,
native controls and audits, with verified remote size/digest. No observations,
capture arrays, residual banks or PIT banks are uploaded. Runtime sources
are pinned in study_manifest.json, code commit2a3e0ca.

The early Windows source-map path lookup failure occurred before any world
query. pipeline_failure.json retains it; only path separator normalization
changed before retry, without refitting or changing policy/data/criteria.
The small known-answer test's initial overbroad expected mask was corrected
before fitting; admissible flat candidates remain admissible. Existing e71
failure and all prior scientific failures are preserved. Reused validation40
is exploratory, and final test41 is unused.
