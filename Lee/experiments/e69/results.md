# Training-state air-budget projection result

All five native/matched comparisons and independent sklearn action/trajectory
audits complete. The prespecified joint improvement criterion fails. Keep e63
as the primary baseline; do not adopt e69 from a favorable secondary contrast.

| Matched-air measure | e63 reference | e69 candidate | Paired difference95% interval |
| --- | --- | --- | --- |
| Air, m3 per23.75hours | 52018.537779 | 51988.734962 | [-49.411557,-10.865370] |
| Violation cost | 21.060125 | 21.167375 | [-.766125,.926250] |
| NH4 violation, percent | 21.844737 | 22.102632 | [-.645000,1.160789] |
| DO violation, percent | .323816 | .178816 | [-.312368,-.018421] |

All actual air matches pass the unchanged tolerance. Mean cost increases
by.107250 and NH4 by.257895 points, with both difference intervals crossing
zero. DO difference-.145000 points passes its nonincrease condition. Neither
a DO-only gain nor uncertainty crossing zero establishes the requested joint
improvement. The same10000 paired policy-seed bootstrap samples and RNG2026
are used; dates, model members and noise draws are not policy seeds.

Calibration matches e63's mean native action on fitting states to at most
2.22e-16. All held independent subsets, strict config reloads and full held
zero-multiplier identities pass. But actual candidate native air averages
58086.334473, versus e63 native55039.345248, a difference3046.989225m3.
For seeds0,2,4 the increases are about8.89%,12.28%,10.77%. Seed1 decreases
slightly and seed3 increases slightly. An exact logged-state mean constraint
does not ensure a closed-loop air constraint or a quality gain.

native_air_diagnostic.json preserves the fitting, held static-equivalent and
actual native changes separately. Static equivalents convert means on logged
states; they are not executed episode volumes. Different policy occupancies
and date/state shifts may matter, but those differences do not identify the
cause. e70 separately preregisters two unchanged-policy replays and an exact
air decomposition to investigate this failure, without retuning e69.

## Secondary mechanism contrast

projection_control_contrast.json verifies that e67 and e69 use the same NNs,
horizon4 rate models, nine-point grid, radius0.5 and primary e63 references.
Both controllers match to the common e63 air targets; they are not separately
matched directly to one another. Their actual volumes are within the
original tolerance, and all complete independent audits pass.

Relative to the unconstrained e67 control, e69 cost difference is-.563125,
interval[-1.179625,-.185000]; NH4 difference-.638684 points, interval
[-1.378947,-.195526]. DO difference+.045921 points has interval
[-.000658,.135526]. This supports a partial mechanism change relative to
e67, not the primary e63 improvement claim or a full Pareto improvement.

The zero-multiplier e67 native trajectory and the zero-multiplier constant-risk
NN null reproduce exactly in the full seed0 smoke. Every final native/matched
replay independently recomputes both shadow-price choices with original sklearn
regressors, their fixed convex interpolation, input preservation, bounds and
executed-air accounting for152000 actions per mode and policy seed.

artifacts.json binds the remotely verified immutable private draft archive:
unchanged base/rate weights, fitted projection configs/traces, all primary
queries, native/matched outputs, audits and logs. It excludes observations,
captures and PIT arrays. The secondary contrast and follow-up diagnostic are
separate compact Git records produced after that primary archive, not claimed
to be inside it. Source data and frozen environment dependencies require
their prior authenticated restoration. No additional environment training
steps or NN/rate fits belong to e69; projection parameters alone are fitted.

Reused validation remains exploratory. Fresh-seed preparation in e68 is not
a successful candidate replication and cannot rescue this failed block.
Final test41 is unused. No physical plant safety or causal improvement is
established by these simulator findings.
