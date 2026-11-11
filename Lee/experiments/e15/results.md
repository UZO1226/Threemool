# Lee/e15 results

All values are validation simulation results. Test remains reserved for team comparison.
Air volume covers 95 intervals (23.75 hours) and is not measured electricity.

| Method | Air (Nm3) | Cost | NH4 violation (%) | DO violation (%) |
| --- | --- | --- | --- | --- |
| ppolag | 66221.520 +/- 16248.055 | 5.870 +/- 1.387 | 5.222 +/- 1.651 | 0.957 +/- 0.258 |
| ppolag_fcsrl | 70594.292 +/- 23785.818 | 9.218 +/- 5.783 | 8.814 +/- 6.494 | 0.889 +/- 0.667 |
| ppolag_fcsrl_curriculum | 61614.877 +/- 11795.705 | 12.000 +/- 5.457 | 12.211 +/- 5.740 | 0.421 +/- 0.056 |

## Paired extension minus FCSRL

- air_m3: -8979.4151, paired bootstrap 95 percent interval [-21795.1662, 4243.6070].
- cost: 2.7820, paired bootstrap 95 percent interval [-3.4940, 9.0580].
- nh4_viol_pct: 3.3968, paired bootstrap 95 percent interval [-3.7653, 10.5589].
- do_viol_pct: -0.4684, paired bootstrap 95 percent interval [-1.0811, 0.0874].

Mean criteria met: False. Air change: -12.72 percent.
Both cost and NH4 intervals below zero: False.

Five seeds and a shared imperfect simulator cannot establish real-plant improvement.
Validation was used for environment selection, so these are exploratory comparisons.
Selected-checkpoint secondary results and every seed are retained in summary.json.
Execution commit: c1c50f923ee495f78b7c71e868096fbc79738bd2. See manifest source and artifact hashes.
