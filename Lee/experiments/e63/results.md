# Risk-only hard-region correction passes the fixed simulation criterion

The complete five-policy-seed study passes all four original matched-air
conditions after independent native/matched action and trajectory audits.
This is evidence in the unchanged e42 research surrogate and reused40-day
validation set. It is not independent confirmation or real-plant safety.
No final test is evaluated and no failed environment candidate is adopted.

## Matched-air primary comparison

| Metric per23.75-hour episode | Original FCSRL mean | Risk-only mean | Paired mean difference | Paired five-seed95-percent interval |
| --- | ---: | ---: | ---: | --- |
| Air, m3 | 52020.577481 | 52018.537779 | -2.039703 | [-31.364000,22.886419] |
| Summed violation-step cost | 22.464125 | 21.060125 | -1.404000 | [-2.204125,-.713375] |
| NH4 violation frequency, percent | 23.208684 | 21.844737 | -1.363947 points | [-2.231184,-.549342] |
| DO violation frequency, percent | .437763 | .323816 | -.113947 points | [-.200526,-.052105] |

All five actual air matches pass, with maximum relative error below.086 percent.
Cost, NH4 and DO are lower for every policy seed, although seed3's NH4 change
is small. The original10000 paired policy-seed bootstrap uses all seeds0..4;
no seed, checkpoint, coefficient, horizon or matching offset is chosen using
quality. Sample SDddof1 and every seed's native/matched values are retained in
comparison.json. Members, draws and dates are not treated as policy seeds.
Violation-step cost is a sum of threshold exceedances, not a monetary cost.
Air is a volume proxy, not measured electricity consumption.

## What changed and what stayed fixed

Keep the same full401280-step FCSRL checkpoints and own horizon4 rate models
for each of the five policy seeds. The model predicts mean NH4/DO indicator
rates over up to four within-day steps; it does not provide certified safety
probabilities. The original grid[0,-.125,.125,-.25,.25], hard radius, clipping,
public observation, model fit, physical costs and environment remain fixed.
Only remove the squared-action-change penalty from the score, retaining stable
first-candidate ties. No regressor fit, extra PPO step or environment-training
transition is added by e63. Learned components and original policy training
were already completed in e55/e56 and their provenance is preserved.

The specification is6b8df22, core/checks4b368fb, comparison/audita337095 and
full-study runnera5f0d7b. All five held training-state identity checks pass,
including exact coefficient1 reconstruction and independent coefficient0
sklearn choices. The complete seed0 native smoke passes flat-risk identity,
config reload and executed-air accounting before the full study launches.

Every full-study seed preserves original baseline replay, independent model/
base/config native replay, exact checkpoints and actual air-only matching.
The five independent audits each reconstruct152000 native and152000 matched
choices using sklearn, complete physical trajectories and executed-air volume.
The final e44 report and same-base/rate-model secondary contrast both complete.
Total runner time is1079.796 seconds. No e63 process remains live.

## Native versus matched policy and secondary contrast

The native wrapper uses55039.345248m3 on average, about5.80 percent more air
than baseline, with cost19.079000, NH419.734342 percent and DO.348816 percent.
It is an air/quality tradeoff under the original5-percent convention, not a
native efficiency improvement. The successful primary statement applies to
the separately budget-matched policy, whose constant offset is applied after
the wrapper and changes actual previous-air feedback. Every native result and
all quality-blind matching queries remain preserved.

Against the completed original horizon4 penalty1 controller at the same
reference air, penalty0 lowers cost by1.362000 with interval[-2.110250,-.722000],
NH4 by1.397763 points with interval[-2.191316,-.709868] and DO by.035921 points
with interval[-.049211,-.021842]. penalty_contrast.json verifies exact same
base/rate models, original baseline and mutual air tolerance. This secondary
comparison does not replace the primary FCSRL comparison or identify a plant
causal mechanism. Matching offsets differ, so it assesses the score change
under the fixed budget-matching protocol.

## Scope, failures and reproduction

All earlier failed controls and environment families remain in the repository.
e63 is the first of12 completed extension/control reports to pass the complete
simulation criterion. Reusing validation across many hypotheses creates
selection uncertainty not removed by a five-seed bootstrap. No fresh policy
training seed or independent date test is claimed. The team test41 remains
reserved for later agreed evaluation. Native/matched mean costs remain above
the12-step limit, and improved violations do not make the controller safe.

The fixed e40/e42 environment passed its original point-error, frequency and
air-response checks, but DO onset/ranking, persistence-relative probability
error, seasonal extrapolation and causal action identification remain unresolved.
Environment effects and this agent effect are separated by an unchanged freeze.
Do not attribute this result to e58/e61, certify plant safety, or treat an agent
win alone as proof that every environment-safety requirement is complete.

lee-e63-risk-only.zip preserves necessary unchanged final base/rate weights,
controller configs, full comparisons/queries/audits/logs, original provenance
and secondary references. It is14,263,028 bytes, SHA256
183c56ee7c864a04adbd337929040d557c37627680cbdb030a198c4443963aa3,
remotely verified in policy draft443333191, asset753857743. artifacts.json
records each file checksum. Original observations, captures and stochastic
banks are excluded. README.md and DARROW_REPRODUCTION.md give exact commands.
