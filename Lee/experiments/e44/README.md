# Lee/e44: Air-only matching and paired policy analysis

This is the evaluation procedure prespecified in e42/e43, not another trained
extension. Compare each final candidate policy with its paired original FCSRL
seed in the exact same frozen environment and common stochastic paths. Report
native behavior first. For the matched diagnostic, add an offset to clipped
deterministic policy actions and clip again to[-1,1]. This changes the policy
being evaluated and is not evidence that its unadjusted behavior dominates.

The selector receives only target mean air, native mean air and a scalar
air-only callback. It has no cost, reward, NH4 or DO inputs. The initial grid
is[-2,0,2]. Saturation endpoints are analytically0 and213750m3 over95 steps:
after clipping, every action is zero or maximum air irrespective of state.
The native air value supplies offset0. Choose a bracket using air only and
bisect for at most25 new queries. Stop at.1 percent relative air error or1m3,
retaining the closest air value, smallest absolute offset and signed offset
as deterministic tie rules. No monotonic policy-response assumption is needed
to retain a sign-changing bracket. Discontinuities may prevent exact matching.
Only label a result matched if within1 percent, as specified before outcomes.
No extrapolation or water-quality-based offset selection is allowed.

The test script checks nonlinear and nonmonotone response, analytic endpoints,
an unattainable discontinuous target, query counts and invalid callback data.
It does not evaluate actual policy performance. After selecting an offset,
re-evaluate and report its separate quality metrics, air mismatch, native
metrics, all policy seeds and paired10000-resample seed bootstrap intervals.
Policy seeds, model members, noise draws and dates are distinct uncertainty
units. Preserve final test41 and the limitations of the repeatedly reused
validation40 and frozen surrogate.

The primary simulation criterion requires all five seed pairs to match air
within1 percent, paired95 percent seed-bootstrap upper bounds strictly below
zero for total cost and NH4, and a DO upper bound no greater than zero. The
report independently recomputes air mismatch rather than trusting a matching
flag. Native and adjusted results, sample SD with ddof1 and every seed remain
visible even when the criterion fails. No test data or intermediate-policy
selection enters this report. A passing simulation criterion is not an
independent real-plant performance claim.

evaluate_pair.py loads model_final only, verifies full401280-transition budgets,
matching seed/configuration, frozen inputs and training-source hashes. Its
native checkpoint replay must reproduce the saved final validation metrics
when evaluated in its training environment. Different-model evaluations are
explicit cross-model sensitivities. native_eval.py stores reusable native
evaluations with checkpoint and freeze identities so the same baseline need
not be recomputed for every comparison. Query trajectories remain outside Git;
compact summaries and scripts are committed. Smoke evaluations are explicitly
marked and rejected by the five-seed report.

The primary capacity contrast compares both two-head variants after each is
matched to the same original FCSRL seed's native air. Reuse the two primary
pair outputs, verifying identical baseline checkpoint hashes, targets and
common stochastic cases. This separates supervision identity at a shared
reference air. The optional kind=capacity comparison instead uses the
capacity policy's native air as its target and is a distinct secondary probe;
it must not replace the common-reference contrast after seeing its outcomes.

The first formal matched pair uses completed seed0 final policies:

```powershell
python Threemool/Lee/experiments/e44/run_pair_batch.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-roots results/e43_first_pair/new_ppolag_fcsrl_split --seeds 0 --out results/e44_split_first --parallel 1
```

This single pair cannot support the five-seed criterion. Continue seeds1..4
in a separate fresh output, then supply exactly five summary.json files to
report.py. capacity_contrast.py takes five split and five capacity primary
pair files matched to the same FCSRL reference. It rejects mixed references
and independently checks both reference and mutual air differences.

The completed first seed fails the matched-air comparison: FCSRL cost24.121875
versus split28.468125, with air47194.44 versus47206.56m3, a.02568 percent
mismatch. NH4 rises25.35855 to29.96645 percent; DO falls.03289 to0 percent.
The native split policy uses84893.57m3 and cost12.1325, so its lower native
cost occurs at substantially greater air. See split_seed0.json for exact
checkpoint identities and the quality-blind matching trace. This negative
single-seed result is retained; remaining seeds and the capacity control are
still required before a method-level judgment.

Independent action replay reproduces that failed pair exactly. Zero-air
fractions are.36542 for FCSRL and.35470 for adjusted split, and mean absolute
step air changes are341.17 versus313.10m3/h. The candidate's matched failure
therefore is not evidence of more frequent shutdown or sharper action changes
than this baseline. The offset clips35.47 percent of candidate commands, but
that observation alone does not establish a causal explanation. Conditional
state/action statistics use different resulting trajectories. See
split_seed0_action_diagnosis.json; no policy or matching rule was changed.

The capacity-control seed0 also fails: matched cost28.294375 versus24.121875,
NH4 percentage29.74211 versus25.35855, DO.04145 versus.03289, with air47157.57
versus47194.44m3. These two negative single-seed comparisons remain separate
from the pending five-seed reports. See capacity_seed0.json.

The four remaining split seeds are evaluated in results/e44_split_remaining
with two workers. Capacity seed1 is evaluated in results/e44_capacity_seed1;
seed0 is already in results/e44_capacity_first. Do not duplicate these pairs
when capacity seeds2..4 complete. Use a fresh later output for those seeds.
