# e75 auxiliary forecasts fail admission, no controller evaluated

All five fixed next-DO model fits complete on243200 fitting transitions,
72960 separate calibration transitions and85120 held transitions per actual
policy seed. Only seed3 passes every prospective admission condition.
The all-five condition fails. The conditional controller is not implemented
or evaluated, and no favorable subset, seed-specific fallback, relaxed
coverage threshold or validation result rescues this study.

| Seed | Model DO MSE | Persistence MSE | Upper coverage | Model FN | Persistence FN | Next-DO events | Admitted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | .004826180 | .005329406 | .949436090 | 32 | 38 | 1247 | No |
| 1 | .003798199 | .004234938 | .951562500 | 41 | 4 | 86 | No |
| 2 | .005989391 | .005819623 | .955227914 | 38 | 73 | 2684 | No |
| 3 | .003360356 | .004221566 | .950046992 | 35 | 48 | 1652 | Yes |
| 4 | .004033847 | .004211207 | .954617011 | 39 | 32 | 1149 | No |

Seed0 fails the fixed.95 coverage condition despite better MSE. Seeds1,2
and4 do not prove lower MSE than persistence under the paired training-date
interval rule. Seeds1 and4 also miss more next-DO violations than calibrated
persistence. In seed1, coverage above.95 coexists with41 of86 violations
missed versus4 for persistence. Global interval coverage and favorable mean
error do not establish accurate rare-event detection. All five beat the
fitting-mean constant, but that weaker control does not rescue persistence
failures. Preserve all onset diagnostics and intervals in summary.json.

audit.json independently reconstructs the full admission decision, physical
metrics, calibration margins, event/onset counts and paired-date intervals.
It also checks original versus compiled unclipped raw predictions on every
401280 training row per seed, plus10000 held action-order probes. The fitter
checks clipped physical means; the independent audit supplies the explicit
unclipped raw check. The audit passes mechanically while admission fails.
No new environment query, policy training interaction or test41 evaluation
occurs. The new offline200-iteration fit per seed and CPU time are recorded
separately from the unchanged401280-step RL budget.

This diagnoses a simulator-trained agent-side forecast, not a new environment
adoption or repair of real DARROW safety timing. Update blocks are separated
for fitting/calibration/diagnosis but their training dates overlap, and the
base policy has trained on all66 updates. Reuse in later model development
must be disclosed. Absolute concentration fitting can lose a strong current
DO baseline, but this study does not establish a causal failure mechanism.
Future residual forecasting is a new hypothesis, not a change to these gates.

## Delivery and reproduction

All five rejected/admitted model weights, margins, fit provenance, runtime,
admission records and full independent audit are preserved in existing
private draft443333191, asset757909078, lee-e75-next-do-forecasts.zip,
788600 bytes, SHA256
83f0f66bcbfe0a2e75ae00ac98f057018fb6ebb1ac03d03a0472d9947fd3ac36.
artifacts.json binds every stored file and verified remote size/digest.
No observations, captures, residual samples or PIT banks are uploaded.
Original NN and frozen-world dependencies remain in their referenced archives.

Fitting starts ataceaa3c, independent audit atd59795b. Exact hashes and runtime
versions are retained in the manifest. In the pinned threemool environment
with numerical threads1, use new output paths and run:

```powershell
python Threemool/Lee/experiments/e75/check_forecast.py --out results/e75_forecast_check.json
python Threemool/Lee/experiments/e75/fit_forecast.py --workspace C:/Users/user/Downloads/IoT_project2 --out results/e75_do_forecast
python Threemool/Lee/experiments/e75/audit_forecast.py --workspace C:/Users/user/Downloads/IoT_project2 --study results/e75_do_forecast --out results/e75_do_forecast/audit.json
```

The fitter reads the committed forecast_check.json; source changes require
regenerating and reviewing that certificate. Existing outputs are immutable.
Locally regenerate authenticated captures for this audit if starting only
from released model weights. Do not remove guards to skip missing captures.
