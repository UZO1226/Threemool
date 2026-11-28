# e74: Extend the authenticated matched reference directly

Latest outcome: the complete original block and all independent audits pass
mechanically, but the joint scientific criterion fails DO with a positive
interval upper endpoint. See [results.md](results.md). Preserve every seed
and do not trigger exposed-block confirmation or change the criterion.

Prospective hypothesis after completed e73 failure,2026-11-17. e73 passes
cost/NH4 but fails DO. Its component guard refers to e63 native, while the
primary comparator is e63 after a fixed air-only offset. The final matched
action can violate that native-relative guard. This is a descriptive
mismatch, not an established causal account of all failures. Native DO and
state shift remain unfavorable, and e71 replication failure stays preserved.

Change the extension's anchor to the actual authenticated matched e63 policy.
For each current obs11, compute the original five-candidate horizon4 risk-only
e63 action, then apply that seed's frozen authenticated offset and clipping.
This is the reference action a_ref. The existing reference's air-derived
offset is reused as an immutable controller parameter, not chosen using
candidate quality. No future observation, additional sensor, member/noise
identity or reference counterfactual trajectory enters the policy.

Center the same nine offsets[0,-.125,.125,-.25,.25,-.375,.375,-.5,.5] on a_ref,
with radius.5 relative to that reference and legal clipping[-1,1]. The radius
is now relative to the true reference, not the original NN. Treat column0 as
the reference; do not minimize the first five new proposals to redefine it.
Require both predicted horizon4 rates no higher than at a_ref plus1e-12.
Replace inadmissible columns with a_ref and its summed rate. Eligibility is
a strict admissible summed-rate gain over column0 by more than1e-12.
Return a_ref exactly outside eligibility or in reference-only mode.

Keep original final401280-step NNs, unchanged horizon4 heads, frozen e42
world and obs11. Fit the original two shadow prices and convex interpolation
weight on eligible fitting updates1 through52 to preserve their a_ref mean.
Use the same[-2^26,2^26] bounds,64 bisections and first-column tie rule.
Require overall fitting mean error at most1e-10. No eligible fitting rows
means reference-only for that seed; never discard seeds. Held updates53
through66 fit nothing. This is a coherent anchor change, not an isolated
ablation of each resulting geometry/calibration implication.

The extra e44 air-only matcher remains unchanged and still adds a constant
offset after this wrapper. The guard therefore remains pre-offset, although
its reference is now the actual matched baseline. Report final executed
predicted changes separately. Do not clamp, cancel or quality-select the
matching offset, expose it as another policy observation, or claim physical
safety. Learned risk plateaus and recursive feedback can still be wrong.

Before quality evaluation require synthetic reference/exclusion/interpolation
and strict config tests, all five held bounds/guard/mean checks,1024 original
sklearn decisions per seed and exact equality with independently loaded e63
reference actions. The full native reference-only controller must reproduce
authenticated e63 matched trajectories exactly. Under constant heads it
must reproduce the original NN plus the same frozen reference offset, not
the unshifted NN. Verify full seed0 control trajectories and report the
changed null target explicitly.

Run all original policy seeds0 through4, all native and actual-air-matched
outputs and independent action/trajectory/air audits. Primary comparator
stays authenticated matched e63. Keep every actual-air match and the original
10000 paired policy-seed resamples, RNG2026: cost and NH4 interval upper
endpoints below zero, DO at most zero. Never round failures into success.
Use e73 only as a secondary mechanism contrast, not a replacement reference.

If the original block passes, run the unchanged entire method on exposed
seeds5 through9 and require the same criterion separately. If both pass,
prepare actual training seeds10 through14 for separately declared additional
confirmation before adoption. Neither pooling nor favorable seed/checkpoint
selection rescues a failed block. Existing5 through9 outcomes are exposed
development information, not unseen confirmation. Reused validation40 is
exploratory; final test41 remains unused. Failure is not a reason to finish
the active goal or silently adopt a different environment.

## Implementation and reproduction

All five original policy seeds now complete fitting-only mean calibration,
all85120 held-state component/bounds/fallback checks and1024 original-sklearn
decision probes. Maximum component increase is zero for both heads in every
seed. preflight.json retains the exact source hashes and parameters. Native
smoke and full-trajectory identity controls remain required before all-seed
comparison; no matched quality gain is established by these held checks.

The native smoke and independent152000-action trajectory audit now pass.
The full frozen-offset NN null and reference-only matched-e63 identity
controls also pass exactly. native_smoke_audit.json records the controls and
zero pre-offset component increases. All five original-seed actual-air
comparisons are now launched, with native/matched independent audits still
required. This is not a completed five-seed improvement result.

anchor_policy.py and check_anchor.py implement and verify the fixed offset,
column0 fallback, admissibility, inner-grid eligibility, interpolation and
strict configuration. fit_anchor.py uses fitting observations only and
independently checks all held rows,1024 original-sklearn decisions, serialized
reload and e73's unchanged legacy changed-row count. Initial fitting is
launched atb07a342. Source hashes and individual fitted configurations are
retained in the output, not inferred from the latest branch commit.

Use the threemool environment with numerical library threads1. Restore the
original data and frozen dependencies by hash, and use unused output paths:

```powershell
python Threemool/Lee/experiments/e74/check_anchor.py --out results/e74_anchor_check.json
python Threemool/Lee/experiments/e74/fit_anchor.py --workspace C:/Users/user/Downloads/IoT_project2 --certificate Threemool/Lee/experiments/e64/baseline_authentication.json --out results/e74_anchor_fit
python Threemool/Lee/experiments/e74/compare.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --candidate results/e55_study/policies/new_ppolag_fcsrl/seed_0 --baseline results/e42_baselines/new_ppolag_fcsrl/seed_0 --correction results/e56_study/models/h4/seed_0 --certificate Threemool/Lee/experiments/e64/baseline_authentication.json --preflight results/e74_anchor_fit/preflight.json --projection results/e74_anchor_fit/seed_0 --previous results/e63_study --out results/e74_native_smoke --native-only
python Threemool/Lee/experiments/e74/run_study.py --workspace C:/Users/user/Downloads/IoT_project2
```

The fitter reads the committed anchor_check.json; regenerated sources require
regenerating and reviewing this control certificate before fitting. Existing
complete output directories are immutable. Reproduce in a new workspace with
consistent runner paths. The original-only arithmetic verifier retains its
legacy filename:

```powershell
python Threemool/Lee/analysis/verify_e71_blocks.py --trial e74 --original-only --workspace C:/Users/user/Downloads/IoT_project2 --out results/e74_initial_verified.json
```

The null control independently evaluates the NN converted to float64 before
adding the frozen reference offset, matching the reference trust policy's
action precision. The reference-only control reproduces e63's archived
matched trajectory, not its native trajectory. Both controls and all five
native/matched audits are mandatory before any adoption claim. All final
executed predicted component changes remain disclosed after the extra offset.
