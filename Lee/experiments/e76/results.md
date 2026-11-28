# e76 residual DO forecasts fail all-five admission

Completed 2026-11-29. Retaining current maxDO and learning its change improves
MSE against both calibrated persistence and the fitting constant in all five
seeds under the fixed paired training-date interval rule. Only seeds2 and3
pass every prospective safety prediction condition. The all-five rule fails.
No controller is implemented or evaluated. No thresholds, seed subsets or
mixtures with e75 models are used to rescue admission.

| Seed | Model DO MSE | Persistence MSE | Upper coverage | Model FN | Persistence FN | Next-DO events | Admitted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.003249422 | 0.005329406 | 0.948895677 | 33 | 38 | 1247 | No |
| 1 | 0.002287186 | 0.004234938 | 0.948672462 | 24 | 4 | 86 | No |
| 2 | 0.003598268 | 0.005819623 | 0.951562500 | 55 | 73 | 2684 | Yes |
| 3 | 0.002338051 | 0.004221566 | 0.951539004 | 43 | 48 | 1652 | Yes |
| 4 | 0.002446990 | 0.004211207 | 0.954405545 | 43 | 32 | 1149 | No |

Seeds0 and1 fail the fixed .95 upper coverage requirement. Seeds1 and4
miss more next-DO violations than calibrated persistence. Lower mean error
therefore does not establish reliable violation detection. NH4 controller
improvement and matched-air policy performance are not measured in e76.

The independent audit passes reconstruction, source and checkpoint bindings,
all physical metrics and admission decisions. Original and compiled raw
changes agree on all401280 rows per seed, with zero maximum difference.
Action ordering passes10000 held probes per seed. These mechanical checks
must not be read as scientific admission.

Each seed uses243200 fit,72960 calibration and85120 held transitions from
the same authenticated official training captures as e75. Dates overlap
between update blocks and the base NN saw all66 updates. e76 also reuses
captures exposed in e75. These are exploratory simulator diagnostics,
not an independent observational holdout or formal coverage guarantee.
The frozen e42 world is unchanged. No new environment queries or test41
evaluation occur. Offline fitting runtime is stored separately in each
fit_summary.json and the study manifest.

## Delivery and reproduction

Fitting code commit edd7dbd; audit code commit f74bda9. The model predicts
raw signed change. Inference must add current maxDO and then floor the final
physical forecast at zero. Loading it as an absolute forecast is incorrect.
All five weights, margins, runtime, source hashes and audit are preserved
in the existing private draft443333191, asset805303302,
lee-e76-residual-do-forecasts.zip,803466 bytes, SHA256
281c895beb18719f1b8a1a0dfd0fdc528aa43120c7165222b511d5d392712645.
artifacts.json binds stored files and verified remote size and digest.
Raw observations, captured transitions and PIT banks are excluded.

With the pinned threemool Python environment and numerical threads1,
use fresh output paths and the authenticated local e75 dependencies:

```powershell
python Threemool/Lee/experiments/e76/check_residual.py --out results/e76_residual_check.json
python Threemool/Lee/experiments/e76/fit_forecast.py --workspace C:/Users/user/Downloads/IoT_project2 --out results/e76_do_forecast
python Threemool/Lee/experiments/e76/audit_forecast.py --workspace C:/Users/user/Downloads/IoT_project2 --study results/e76_do_forecast --out results/e76_do_forecast/audit.json
```

The fitter reads committed residual_check.json. Preserve source bindings
and regenerate authenticated captures when needed rather than bypassing
checks. The automatic research goal was cleared. This completes the bounded
e76 request; no new autonomous study is started.
