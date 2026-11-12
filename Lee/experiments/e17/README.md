# Lee/e17: Contractive monotone autoregressive environment

## Reason for the next model family

e16 enforces the desired recursive air ordering but fails its safety adoption
gate. The selected calibration produces NH4 violations 8.51 percent versus
17.32 percent observed, and NH4 MAE improves less than 10 percent. Its raw
cooperative trees produce about 66 to 71 percent NH4 violations. A nonlinear
state feedback can amplify small one-step errors even when it is monotone.
This motivates testing contraction rather than relaxing the e16 gate.

Fit next state as a fixed diagonal autoregression plus an exogenous/air GBM:
next[j] = alpha[j] * state[j] + forcing(exogenous, current_air, previous_air).
The forcing cannot read current concentration. NH4 forcing is nonincreasing
in air; all three DO forcings are nondecreasing in all five air zones. NO3
has no imposed air sign. With common inputs, initial state errors contract
componentwise by alpha, and increasing air has the desired trajectory order.
There is no physical identification claim and removing state interactions
may cause underfit. This is an empirical state-transition model.

## Preregistered comparison

Four combinations: NH4 alpha 0.95 or 0.99, DO alpha 0.2 or 0.8, NO3 alpha
fixed at 0.99. GBM 15 leaves, 200 iterations, learning rate .05, minimum leaf
40, L2 1, five members with fixed 80 percent subsets. Fit the first 158 training
days, calibrate a constant five-output bias on the last 30 training days using
the same bounded 120-evaluation procedure as e16. Keep raw and calibrated
versions. Select from these eight by inner calibration score only.

Audit the 40 development days and apply the unchanged e16 adoption gate.
No new policy training if it fails. The development set is reused exploratory
data, not independent confirmation. Test remains untouched. Verify the fast
prediction path against sklearn and test the recursive contraction bound.
Read e16 and e15 for provenance, units, protocol differences and limitations.
