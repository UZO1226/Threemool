# Forecast admission passes, controller improvement remains unproven

Calibrated at3c86c43 on2026-11-29 and independently reconstructed by
audit_upper.py at4d0cfae. All five seeds pass the unchanged forecast admission
criteria. forecast_summary.json and forecast_audit.json preserve the exact
source and dependency hashes and independently recomputed MSE intervals.

| Seed | Held upper coverage | DO violation misses | False positives | Added alarms versus persistence |
| --- | --- | --- | --- | --- |
| 0 | .983976 | 12 | 3571 | 693 |
| 1 | .985244 | 2 | 657 | 84 |
| 2 | .986948 | 16 | 3944 | 871 |
| 3 | .986924 | 13 | 4171 | 469 |
| 4 | .988898 | 11 | 3335 | 840 |

The e76 point forecast and MSE remain unchanged. Only uncertainty margins
and the persistence alarm floor change. The floor includes every original
persistence alarm, so nonincrease of misses versus persistence follows by
construction. These gates alone do not demonstrate a learned safety gain.
Added alarms and thousands of false positives can limit useful actions.
Every fixed DO stratum has held empirical coverage above .97, including the
previously weak high-current-DO stratum. These reused training captures are
exploratory and provide no independent observational coverage guarantee.

The audit independently reconstructs calibration quantiles, every admission
condition, physical metrics and paired-date MSE intervals. It compares raw
original and compiled signed changes on401280 rows per seed and verifies
action order. No new world model, offline point fit, environment query or
final test41 evaluation is performed at this stage.

The controller stage is in progress. Fitting preserves the matched-reference
mean action on fitting observations and checks original sklearn upper bounds,
component guards and disabled-shield e74 identity on held observations.
Only complete native and matched-air comparisons with independent trajectory
audits can establish the prospective policy criterion. No such gain is
claimed by this forecast report. Later result records must distinguish
forecast admission from controller and replication outcomes.
