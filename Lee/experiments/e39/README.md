# Lee/e39: Saturating implicit Monod removal for NH4

## Distinct nonlinear-state hypothesis

All neural backbones so far are affine in current state, with coefficients
depending on exogenous history and initial context. Fractional linear NH4
depletion can erase high-NH4 episodes too quickly when actual removal capacity
saturates. e33 changes temperature scaling but remains affine in state. Test a
positive learned source and saturating NH4 removal instead of another affine
coefficient correction. This is independent of e38's distributional tree family.

Keep e32's context-conditioned backbone for DO, NO3 and hidden coordinates.
Replace its NH4 row with an implicit Monod update. Let B=a times currentNH4
plus a positive learned source. Let v be a positive learned removal capacity
times oxygen/(Kdo+oxygen), using the newly predicted/clipped mean DO. Solve
x+v*x/(Knh+x)=B for the nonnegative root. The source and capacity depend on
exogenous history and strictly pre-episode context, not future actions. Use
the documented BSM temperature factor for capacity as an informed prior.

Knh and Kdo start at1 and .4, then are learned positive global parameters with
a .01 floor. These initial values come from primary BSM2-Python documentation,
not Tilburg calibration:
https://bsm2-python.readthedocs.io/reference/bsm2_python/bsm2/init/asm1init_bsm1/
The benchmark numeric values are initialization heuristics in the reported
channel units. No unverified NH4-to-N molecular conversion is inferred.
The temperature-factor source is the same as e33. Capacity has units of mg/L
per update; it is not the benchmark growth-rate parameter. This is a lumped
NH4 reaction hybrid, not a full ASM1 model, plant-volume estimate or complete
nitrogen mass balance with the separately learned NO3 output.

The implicit root increases with available NH4 and decreases with removal
capacity. It stays nonnegative, avoids explicit-Euler negative concentrations,
and permits slower fractional clearance at high NH4. Backbone DO retains its
cooperative signs, so increasing air raises modeled oxygen and cannot raise
next NH4 under fixed noise/context. Do not claim global contraction from the
linear backbone's .995 cap: the NH4 row is overridden by nonlinear chemistry.

Initialize the source/capacity so the fitting mean is an equilibrium at its
mean temperature/oxygen. Verify that equilibrium, positive root equation,
saturation, directional signs, learnable reaction heads and independent95-step
Torch/cache agreement before fitting. Noise is added after the reaction,
followed by physical clipping. NH4 calibration bias enters available source;
other biases enter the backbone before modeled oxygen is used. The same
reaction helper is used by recursive simulation and teacher residual-bank
generation; the independent Torch implementation checks that consistency.

Keep original full-day loss and158/30 split, five fitting seeds and80-percent
date subsets, batch16, Adam .001,80epochs,20/40/80 inner selection, original
bias-calibration bounds, four-step innovations and16-scale search. No mixed
teacher loss, tail weighting, new partition or per-episode assimilation. Model
selection uses inner data only. Development40 is exploratory reused audit;
test41 remains unevaluated. Keep every e16 adoption threshold and report NH4
and DO separately. No agents until an environment passes.

Source/data hashes, preparation, original FCSRL commit, CPU packages, physical
units,95-step horizon, reward/cost/action limits follow e24/e32. Five model
fits are not policy seeds. Further failure does not complete the active goal.

```powershell
python Threemool/Lee/experiments/e39/check_monod.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out Threemool/Lee/experiments/e39/monod_check.json
python Threemool/Lee/experiments/e39/monod_hybrid.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e39_monod
```
