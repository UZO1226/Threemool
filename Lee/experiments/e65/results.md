# Nested-grid result versus authenticated e63

The complete five-policy-seed study fails the prespecified joint improvement
criterion. All actual air matches, independent native/matched sklearn action
checks, trajectory replays and air accounting pass. Keep e63 as the primary
baseline. A small cost improvement does not satisfy the NH4 and DO rules.

| Matched-air measure | e63 reference | e65 candidate | Paired difference95% interval |
| --- | --- | --- | --- |
| Air, m3 per23.75hours | 52018.537779 | 52031.636428 | [-14.792957,34.541005] |
| Violation cost | 21.060125 | 21.025750 | [-.076250,-.000875] |
| NH4 violation, percent | 21.844737 | 21.810395 | [-.076579,.001579] |
| DO violation, percent | .323816 | .321974 | [-.005526,.001842] |

Intervals use the unchanged10000 paired policy-seed bootstrap resamples,
RNG2026. Model members, dates and innovation draws do not count as additional
policy seeds. Cost difference is-.034375, NH4-.034342 points and DO-.001842
points. All means decline, but NH4 and DO upper interval endpoints exceed
zero. No alternative seed subset, match offset or threshold is selected.

Native candidate air is55053.478355 with cost19.100000, NH4 19.753158% and
DO .352105%. It uses more air than the primary matched e63 baseline, so that
quality comparison is not an efficiency win. It also does not outperform
e63's separately documented native means: air55039.345248, cost19.079000,
NH4 19.734342% and DO .348816%. Preserve both native and matched results.

Held-state geometry passed for all85120 rows per seed, with1481,3870,2637,
3650,4885 actions changed. Independent sklearn subsets matched exactly.
Lower predicted risk on those states did not establish a joint policy gain.
This bounds the outcome of the fixed17-point grid, not every possible
controller or the causal accuracy of the risk regressors.

## Interrupted runner and exact query recovery

The original runner terminated on a Windows status-file replacement
PermissionError after seeds0,1,3 completed. Its exception handler ended the
remaining processes. Actual process inspection and a terminal failure.json
established this was a stopped job, not an observation timeout. All original
logs and partial outputs remain preserved under results/e65_study.

resume_study.py authenticates every originally pinned source. It copies the
three completed comparisons into results/e65_recovery. For seeds2 and4 it
reuses the original baseline/native files and completed air-only queries,
bound to the original manifest and flushed offset/volume logs. The original
native file is written only after exact independently reloaded native replays.
It invokes the unchanged matcher from its beginning, returning the previously
computed volume for the identical requested offset, and computes only new
offsets. It never chooses a query by quality or changes the target volume.
Recovery provenance records every reused and fresh call with file digests.

Every seed, including the copied and recovered comparisons, then undergoes
a complete independent sklearn action/trajectory audit for native and matched
policies:152000 actions per mode and seed. These pass before the final report
is generated. Status writes in the recovery driver retry transient Windows
file locks. The historical pinned runtime was not edited. The complete
recovered study has no remaining running or pending stages.

comparison.json and audit_seed_0.json through audit_seed_4.json preserve the
compact findings. artifacts.json identifies the verified existing policy
draft asset: unchanged base/rate weights, all original/recovered queries,
configs, terminal failure evidence, provenance and audits. No observations,
training captures or PIT banks are included. The partial-study archive
refusal check passes before the completed archive is built and verified.

This remains reused-validation exploratory evidence from five already
trained401280-step policies. No new NN or risk model was trained, no final
test41 evaluation occurred, and no fresh-seed independent success is claimed.
