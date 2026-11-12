# Lee/e32: Persistent episode context in state forcing

## Hypothesis and prospective ablation

e31's joint local/full-day loss improves NH4 frequency to9.75 percent but fails
the unchanged gate. Existing models use the strictly past four-hour context
only to initialize decaying latent states. A latent initial-condition effect
can vanish as the day evolves, whereas unobserved biological capacity or
operating regime may persist. Test a learned, action-independent context offset
to state forcing throughout the episode instead of a fixed linear persistence
blend of NH4, which failed inner selection in e25.

Add an encoder304 to64 to7, Tanh hidden activation, to the original corrected
e24 model. Its seven standardized forcing offsets depend only on the observed
context before episode start. Broadcast them across the95 future steps. Do not
update the context from future measured states during rollout. No action or
future-quality labels enter this encoder at prediction time. Because the same
offset is held fixed across paired-air probes, orthant monotonicity and the
.995 state-matrix cap are preserved. The offset is an empirical latent regime,
not an identified biomass measurement or a causal process parameter.

Initialize the added output weights/bias to zero, making all95 initial outputs
identical to e24. Restore the constructor's Torch RNG state after initializing
the added module so the base fitting batch permutations remain paired. Test
full-day equivalence, target chronology, and a finite nonzero gradient on the
added head before fitting. The model has extra capacity; a successful result
alone would not identify the unique physical reason for improvement.

Change only persistent context. Keep original full-day loss, not e31's mixed
loss or e30's tail weights; original .995 cap, first158/last30 inner split,
five fitting seeds,80-percent subsets, batch16, Adam .001,80 epochs and shared
20/40/80 checkpoint selection; original four-step innovations and16-scale grid.
Keep e16 adoption thresholds unchanged. Development40 is reused exploratory
audit, official test41 remains unused. Fitting/calibration stats use training
only. Model seed count is not policy seed count.

The reusable cache recognizes explicit episode_coefficients for this class.
Old classes retain their original coefficient path. Audit must load the context
class and rejects a mismatched model kind. Exact Torch/NumPy recursive and
same-noise air-order checks remain required before any environment adoption.

Source/data hashes, CPU packages, original FCSRL commit, mg/L and m3/h units,
95-step horizon, reward/cost/action bounds follow e24. No new policies before
the environment passes. A failure leads to another justified hypothesis.

```powershell
python Threemool/Lee/experiments/e32/check_context.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out Threemool/Lee/experiments/e32/context_check.json
python Threemool/Lee/experiments/e32/persistent_context.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e32_persistent_context
```
