# Lee/e33: Kinetic temperature scaling of NH4 retention and removal terms

## Hypothesis and source

Unconstrained neural extrapolation of temperature has not repaired cold-period
NH4 event forecasts. Test an explicit kinetic prior instead of treating the
latest failed model as an external blocker. BSM2-Python's published ASM1 code
scales autotrophic growth by exp(log(mu_A/.3)/5 times(T-15)), with its documented
reference mu_A=.5. Use kappa=log(.5/.3)/5, fixed prospectively, in an empirical
neural transition. Sources:

- https://bsm2-python.readthedocs.io/reference/bsm2_python/bsm2/asm1_bsm2/
- https://bsm2-python.readthedocs.io/reference/bsm2_python/bsm2/init/asm1init_bsm1/

These are primary implementation references, not calibrated Tilburg parameters.
This candidate is not a full ASM1 mass balance or the actual DARROW process model.

## Transition change

Keep the corrected base neural model and convert its current temperature back
to degrees C using fitting-only normalization buffers saved in the checkpoint.
Let r=exp(kappa times(T-15)). If alpha is its NH4 self-retention coefficient,
replace alpha with alpha^r. Scale negative NH4 cross-state coefficients, negative
NH4 forcing, and its nonpositive air terms by r. Leave positive source terms
and other output rows unchanged. At15C the scaling operator is the identity.
With otherwise fixed coefficients, colder temperature slows NH4 depletion.
The original neural temperature inputs remain, so the entire learned model
is not claimed to be globally temperature-monotone or mechanistically identified.

This explicit prior differs from e19's learned linear temperature extrapolation.
Positive factors preserve all state/air response signs and recursive orthant
ordering. NH4 retention stays between0 and1, but total effective row sums are
not guaranteed below .995. Report actual maxima and stability under clipping;
do not claim contraction. The e16 adoption gate never required a particular
matrix cap, and none of its prediction or safety thresholds is relaxed.

Before fitting check15C coefficient identity, slower depletion under cold
temperature at fixed coefficients, air-direction signs, finite nonzero observed
head gradients and full-day Torch/cache agreement. The class is explicit in
audit metadata to avoid loading weights into a wrong transition rule.

## Fixed conditions

Same first158/last30 training partition, five paired fitting seeds and80-percent
subsets, original full-day loss, batch16, Adam .001,80 epochs, common20/40/80
inner selection, bias calibration, four-step innovations and16-scale grid as
e24. No persistent-context encoder, mixed horizon, tail weights or cap1.05.
Keep scoring SD and adoption thresholds fixed. Development40 is exploratory
reused audit; official test41 is not evaluated. All fit statistics come from
training dates. Original FCSRL code remains unchanged, and these are environment
seeds, not policy seeds. Units/reward/cost/horizon/bounds/hashes/runtime follow
e24. No policy training before an environment passes.

```powershell
python Threemool/Lee/experiments/e33/check_kinetics.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out Threemool/Lee/experiments/e33/kinetic_check.json
python Threemool/Lee/experiments/e33/kinetic_scaling.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e33_kinetic_scaling
```
