# Lee/e22: Multi-step latent state model with causal history

## Prospective question

e16-e20 improve average errors or structural properties but do not pass NH4
safety calibration. e21 shows much better NH4 fidelity with frequent measured
state resets. Test whether missing state/history and one-step fitting contribute
by training a constrained latent state model directly on 16-step sequences.
This is not a claim that hidden biomass has been identified.

Five observed concentrations plus two learned nonnegative latent states evolve
under a sign-consistent, contractive seven-state matrix. The NO3 coordinate
does not feed back into the other coordinates. In transformed standardized
coordinates, matrix entries are nonnegative and row sums below .995. Matrix
gates may depend on current exogenous temperature; NH4/latent persistence
increases toward colder conditions. Forcing is an exogenous MLP and a monotone
log-air term. No current or future logged action is fed to the latent initializer.

Two paired variants use the same architecture and initial parameters: history
1 repeats the latest available context; history 16 uses the past four hours.
The latent initializer uses only earlier state/exogenous/action measurements.
Exogenous forcing may use current and previous exogenous inputs, not future
concentrations. Initial context is fixed and common for paired air interventions.
Latent variables are learned coordinates, not certified physical tank contents.

Fit targets use only the first 158 training dates. Earlier observations may be
used as strictly causal context, including the partial day preceding May 6.
Five fitting seeds use 80 percent date subsets. Each uses 20 fixed epochs,
64-sequence batches, length 16 and stride 4, Adam .001. Loss is equal-weight
normalized MSE/MAE plus .5 times NH4/DO soft-threshold squared errors. No
development labels enter gradients or early stopping. Capacity is identical
across history variants; fitting members are not policy training seeds.

Bias and residual innovations use only the last 30 training dates. Select
history variant by inner proper score, then e18's fixed 16-point innovation
grid. Audit selected and zero-noise forecasts on development; retain the
unchanged e16 adoption gate and 16-draw sensitivity. Verify bounded row sums,
common-innovation action ordering, context causality and numerical finiteness.
Test dates remain unused. A failed model is not used for agent learning.

Model inputs include challenge-provided generated ASM1 influent components.
Their real-time availability is not established. Air is documented m3/h,
NH4/DO mg/L; earlier historical reports using Nm3 or mg N/L do not establish
reference conditions or nitrogen basis. See source review in e35 by Yoo.
