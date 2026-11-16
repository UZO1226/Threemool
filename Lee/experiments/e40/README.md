# Lee/e40: Preserve declared monotone bounds after quantile leaf updates

## Reproduced implementation behavior

e38 forecasts NH4 frequency16.43 percent versus17.32 observed, with four
members inside individual tolerances, but DO2.69 versus1.16 and large recursive
air reversals prevent adoption. Fast inference matches sklearn exactly.
A separate synthetic monotone-constraint regression using installed sklearn
1.9.1 also declines as its constrained input increases: maximum decline .05586,
nine reversals. This is independent of the wastewater wrapper or quantile sorting.

Installed _update_leaves_values replaces nondifferentiable-loss leaf values
after grower.grow with shrinkage times a residual quantile, without clipping
to the grower's previously computed bounds. These updates can violate the
declared monotonic constraints. This is a locally reproduced source behavior,
not an assertion that upstream maintainers have accepted a reported issue.
Primary code: sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
and grower.py in the scikit-learn repository.

## Scoped correction and identical repeat

During this experiment's fitting only, wrap the installed leaf-update function:
run the original update, then clip each leaf to shrinkage times its inherited
ancestral limits. Reconstruct limits from the root and each constrained split's
unshrunk SplitInfo child values. Merely using leaf.children_lower_bound and
children_upper_bound is insufficient: max-leaf/max-depth early returns finalize
children before setting those fields. A first small-data test of that naive
clamp still reverses by .002848, so it is not used for full fitting.
Bounds are unshrunk in the grower;
the leaf prediction is shrunk, so both limits must be scaled. This enforces
the already supplied constraints, not a new validation rule. Installed package
files are not edited. Restore the original function in finally. Pin sklearn
1.9.1 and the original function's source SHA256; fail on an unexpected version.
The serialized trees need no patch at prediction time.

First independently reproduce the old synthetic failure and require zero
reversals after correction, finite predictions, actual leaf clipping and
function restoration. Before long recursive scoring, require one-coordinate
order on observed/perturbed features as well as numeric fast/sklearn identity.
Then repeat e38 exactly: same105 model fits, bootstrap dates, seeds, five
quantiles,200 trees,15 leaves, train158/inner30, PIT handling,20 inner settings,
8 draws and16-draw sensitivity, and original e16 gate. No development tuning
or final test. Record the correction's effect separately from method failure.

The patch preserves current pinball leaf updates and their existing treatment
of regularization, changing only the already intended monotonic bounds.
Clipping each univariate convex leaf optimum into its permitted interval is
feasible; it is not a claim of globally optimal monotone quantile-tree fitting.
The five model seeds remain distinct from policy seeds. Units, runtime,
original FCSRL commit, data hashes, reward/cost/horizon/action bounds follow
e38/e24. No agents before the environment passes.

```powershell
python Threemool/Lee/experiments/e40/check_bounds.py --out Threemool/Lee/experiments/e40/bounds_check.json
python Threemool/Lee/experiments/e40/monotone_quantiles.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e40_monotone_quantile
```
