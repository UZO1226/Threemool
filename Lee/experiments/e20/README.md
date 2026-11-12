# Lee/e20: Coupled contractive monotone state feedback

## Prospective hypothesis

e16 allows NH4/DO interaction but does not bound state feedback. e17-e19
bound feedback but remove interactions. Test whether retaining sign-consistent
NH4/DO coupling under an explicit contraction bound improves safety fidelity.
This is another state-transition model family, not a new policy or a relaxed
adoption threshold. No agent is trained unless its environment passes.

Transform concentration coordinates with training standard deviations and
sign vector (-1,+1,+1,+1,+1). Fit a nonnegative state matrix with row sums
at most rho in the NH4/DO core. NO3 does not feed back into this core and
has only its own autoregression. rho choices .95 and .995 are fixed before
outcomes. Estimate each matrix row using a convex quadratic fit with
exogenous and air controls, nonnegative transformed air coefficients and
ridge penalty .001. Then fit the same 15-leaf/200-iteration monotone GBM
to the remaining exogenous forcing. State columns are excluded from trees.
Fit only the first 158 dates; five members use fixed 80 percent subsets.

This construction enforces recursive air ordering and weighted infinity-norm
initial-state contraction for fixed exogenous inputs and actions. It is a
prior with empirical coefficients, not a validated stoichiometric plant model.
An independently changing controller can still have closed-loop instability.

Train-only five-output bias calibration uses e16's bounded procedure. Choose
between the two calibrated matrices solely by inner calibration score.
Then select e18's 16 innovation scale combinations by its proper inner score.
Audit only that selected stochastic model and the selected deterministic
reference on development. Verify full fast/sklearn equality and weighted
contraction. Report 16-noise-draw sensitivity. Apply the unchanged e16 gate.
No test dates are evaluated. Preserve failures and all inner comparisons.
