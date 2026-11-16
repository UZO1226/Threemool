# e62: Calibration-only decomposition of context noise channels

e61's primary basic evaluation fails NH4 frequency, and its primary independent
donor/full-day/air-order replay passes while the remaining audit is running.
Before another candidate, distinguish changing the NH4 PIT path from changing
the DO PIT path and subsequent state coupling. This is an internal-model
diagnostic, not a repair or adoption of e61 and not a plant counterfactual.

Freeze e51's existing weights, alpha.25, five members, eight draws and all30
calibration dates. Use e61's exact global29-day leave-self-out and context-five
whole-path samplers, with its fitting-only six-feature scaling. Cross NH4 noise
source global/context with joint DO1/DO2/DO3 source global/context, four fixed
conditions. NO3 remains.5. Copy each channel's complete95-step path; keep the
three DO channels together. Mixed conditions deliberately change NH4/DO joint
dependence, which limits any physical or probabilistic interpretation.

Before interpretation, global/global must reconstruct e61's stored global
calibration scores and context/context must reconstruct its context scores.
Authenticate data, models, sources and bank. Report every condition's MAE,
NH4/DO frequency/Brier/AP/AUC/onset0/1/4 and paired circular five-day Brier
intervals versus global/global. No subset, tail, k, alpha, policy seed or
criterion search. Calibration already estimated the banks and is not held-out.

No validation or final test evaluation, model fitting, new agent training,
environment freeze or adoption decision is allowed in this diagnostic. Use
the complete outcome to motivate a later prospectively specified candidate.
Keep source observations, bank and trajectory arrays local; publish aggregate
summaries and identity checks only. Preserve e61 failure and its full audit.

Specification and implementation commit8f41b0e precedes all outcomes. Launch:

```powershell
python Threemool/Lee/experiments/e62/decompose_channels.py --data kaggle_data/darrow_15min.pkl --e51 results/e51_blended_pit --e61 results/e61_context_sampling --preflight Threemool/Lee/experiments/e61/preflight.json --out results/e62_channel_decomposition
```

Use pinned threemool Python, CPU, DARROW_FAST_DYNAMICS=1 and OMP/OPENBLAS/MKL/
NUMBA thread counts1. Two independent inference tasks run concurrently, with
no model fitting. Existing output directories are rejected. Each completed
condition writes an aggregate report; summary.json requires all four and both
e61 identity controls to finish. Inspect actual process liveness before
resuming. No completed outcome is recorded yet.
