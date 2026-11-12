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
