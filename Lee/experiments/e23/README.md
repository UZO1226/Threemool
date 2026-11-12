# Lee/e23: Full-day latent-model training and convergence control

## Prospective design

e22's selected history-16 model improves NH4 MAE but misses event frequency,
has poor NO3 replay, and is trained for 16 rather than 95 recursive steps.
Test full 95-step fitting before declaring that history-conditioned latent
dynamics cannot meet the environment gate.

Keep e22 architecture, causal history 16, normalizations, member date subsets,
loss, Adam .001, and state/action priors. Train from scratch on complete
95-step days for 80 fixed epochs, batch size 16 days. Save ensembles at
20, 40 and 80 epochs. Bias-calibrate each ensemble on the same 30 inner
training days and choose a common checkpoint epoch by inner proper score.
No per-member checkpoint selection on development. Then select the unchanged
e18 16-point innovation scale grid on inner data. Audit only that selected
checkpoint/scale and report 16-draw sensitivity under the unchanged e16 gate.

This jointly changes fitting horizon and optimization budget; it is not an
isolated one-factor performance attribution. All outcomes, learning curves,
checkpoint checksums and calibration results are retained. No development
gradient/early stopping, no test evaluation, no agent learning on a failed
environment. The latent variables remain empirical unobserved coordinates.
