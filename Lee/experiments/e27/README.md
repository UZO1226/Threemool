# Lee/e27: Test whether enforced contraction erases NH4 risk memory

## Prospective hypothesis

e26 improves NH4 Brier slightly but still predicts only 6.86 percent violation.
Its recall at .5 probability is .202, with 525 missed development event steps.
All earlier neural models enforce a standardized state-matrix row sum below
.995. This prior may cause excess mean reversion and prevent persistent risk
from the observed initial state under seasonal shift. Test a row-sum cap 1.05,
which permits slow amplification while preserving orthant monotonicity and
the existing physical output clipping. This is a state-response hypothesis,
not a relaxation of the environment adoption gate.

Change only the gate cap. Reparameterize its initialization so zero-temperature
state coefficients exactly match e24's initial coefficients. Verify equilibrium
and nonzero gradients as well as coefficient equivalence at initialization.
All air-gain signs, NO3 repair, architecture, loss, optimizer, five paired
fitting seeds, 80-percent date subsets, full95-step training, 80 epochs and
shared20/40/80 inner checkpoint selection are unchanged. No e26 noise change
is combined here: use e24's original four-step innovations and 16-scale grid.
This isolates the matrix-cap change. Record whether learned matrices actually
exceed1; if not, the proposed mechanism is not exercised by the fit.

First158 training dates fit, last30 select/calibrate, development40 audits only.
Test41 remains unused. Reuse development labels only for exploratory research,
never call it fresh confirmation. Preserve e16 adoption thresholds and full-day
monotone air-order checks, with 16-draw sensitivity. Report NH4 and DO separately.
Five fitting seeds are not policy seeds; train no new policy on a failed model.

The reusable e23 runner accepts a model/cache factory, with unchanged defaults
for earlier experiments. e22 cache validation accepts an explicit cap, default
.995. e27 does not claim contraction for its larger cap. Recorded values are
mg/L and m3/h, with unchanged reward, cost, horizon and bounds from e24. Data
hashes, original FCSRL source commit, package versions and CPU threads follow
e24 and e15/runtime.json. Execution commit is captured at run start.

```powershell
python Threemool/Lee/experiments/e27/check_initialization.py --data kaggle_data/darrow_15min.pkl --out Threemool/Lee/experiments/e27/initialization_check.json
python Threemool/Lee/experiments/e27/slow_memory.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e27_slow_memory
```
