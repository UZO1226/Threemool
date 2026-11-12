# Lee/e24: Repair dead NO3 initialization and repeat full-day fitting

## Discovered implementation defect

The neural model initialized all 70 air logits to -5 for small positive
softplus gains. NO3 is the one free-sign channel and uses raw logits rather
than softplus, so its actual initial gains were -5 while its equilibrium
forcing initialization assumed zero. Its prediction became strongly negative,
was clamped to zero and had zero NO3 head gradient. This explains the exactly
identical NO3 MAE in e22 and e23. It is not a dataset limitation.

The minimal fix initializes the ten free-sign NO3 air coefficients to zero.
All nonnegative gains retain -5 softplus initialization. Verify the initial
equilibrium against the fitting-data mean and nonzero output gradients. A
control copy with the former logits must reproduce the dead head. Keep old
results labeled historical and invalidate their model-family interpretation.

## Repeat design

Repeat e23 without other changes: history16, five fitting seeds, 80-percent
date subsets, complete 95-step sequences, Adam .001, 80 epochs, batch16,
same loss, priors and normalizations. Select a shared checkpoint at 20/40/80
on inner proper score after bias calibration, then the original 16 innovation
scales on inner data. No development tuning, no test evaluation. Retain the
unchanged e16 environment gate and 16-draw sensitivity. Do not train a policy
on a candidate that fails. The new run uses results/e24_fixed_initialization.

The one initialization correction also affects shared training gradients.
It is not a promise that NH4 calibration will be repaired automatically.
