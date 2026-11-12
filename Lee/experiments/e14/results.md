# Lee/e14 results

Validation uses 40 days, five ensemble members and 95 recursive transitions.
The initial measured state is excluded from error and violation rates.

| Model | Normalized rollout MAE | NH4 rollout MAE | NO3 rollout MAE | DO1 MAE | DO2 MAE | DO3 MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Old delta GBM | 0.9660 | 2.4823 | 1.6458 | 0.4518 | 0.3058 | 0.2706 |
| Persistence | 0.9322 | 2.3361 | 1.0560 | 0.4749 | 0.4934 | 0.1122 |
| Direct 7 leaves | 0.8380 | 1.9448 | 1.8075 | 0.4154 | 0.2304 | 0.1797 |
| Direct 15 leaves | 0.8549 | 2.1282 | 1.6743 | 0.4138 | 0.2693 | 0.1609 |

The preregistered rule selects direct 7 leaves. Relative to old GBM it reduces
normalized recursive MAE by 13.25 percent and NH4 MAE by 21.65 percent. NO3
MAE worsens 9.82 percent, close to the prespecified 10 percent rejection limit.
This is an exploratory candidate, not an established accurate plant model.

The old model's reported R2 of 0.24 for NH4 measures change prediction, not
concentration prediction. Next concentration R2 is 0.965, compared with 0.953
for persistence. Its single-step NH4 MAE is 0.279 versus persistence 0.317.
However, recursive NH4 error grows to 2.482, worse than persistence 2.336.
High single-step level R2 alone therefore gives misleading reassurance.

Constraint calibration remains weak. Direct 7 predicts NH4 violation at
26.98 percent versus actual 17.32 percent. DO violation is 0.17 percent versus
actual 1.16 percent, worse calibrated than old GBM (1.80 percent). Improving
average error has not solved safety calibration. Simulation safety claims
cannot be transferred to the actual plant or described as causal validation.

Increasing current air by 50 percent of recorded air while holding the same
state and previous air fixed lowers mean predicted NH4 and raises DO in all
models. The direction is imposed as prior knowledge. It does not identify
the real causal response. Persistence has zero action response and cannot
serve as a controller training environment.

Additional fixed-air trajectory checks are in action_response.json. With
recorded air scaled from 0.5 to 1.5 and clipped to the action range, direct 7
reduces mean NH4 from 6.89 to 1.09, but mean DO3 also falls from 0.359 to 0.265.
Thus the single-step DO monotonic prior does not carry through recursive
cross-state dynamics. This unresolved behavior is another reason not to
treat simulated policy gains as demonstrated process improvements.

Training-only replay gives daily cost 10.74894, rounded to limit 11. The same
limit is used for every e15 method. This differs from protocol v1's old-model
limit 12, so e15 does not replace official team results. The old calibration
helper enumerates all splits; an attempted invocation was interrupted without
producing a report. The new calibrate_train.py explicitly evaluates only train.
No test result was inspected or used in selection.

## Inference equivalence

Optional DARROW_FAST_DYNAMICS=1 enables finite-input numerical tree inference
with Numba. It rejects categorical trees and nonfinite inputs. It is off by
default and retains sklearn inference as fallback. For old GBM and both direct
candidates, predictions match sklearn exactly on all 3800 validation transitions,
all five ensemble members, and mixed-member batches (maximum absolute error 0).
See fast_equivalence.json. This optimization changes computation, not the model.

fast_equivalence_stress.json additionally checks 512 perturbed state and air
inputs for every model. The selected model's small-batch inference is about
49 times faster in that local timing; training speedup is smaller and is not
a learning-method improvement.

## Reproduction and provenance

Audit model code commit: b9fefe5. Data and model hashes are in summary.json.
Python 3.11.15, torch 2.15.0+cpu, scikit-learn 1.9.1, Windows x64, CPU.
Data source and upstream FCSRL license are documented in Lee/code/darrow/README.md.
Train 188 days, val 40 days, test 41 days, chronological split from e07.
Models and logs are local at results/e14_audit; raw data is not redistributed.

Run from the parent directory of Threemool with the project Python environment:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python Threemool/Lee/experiments/e14/audit.py --data kaggle_data/darrow_15min.pkl --old kaggle_data/darrow_surrogate_gbm.pkl --out results/e14_audit
python Threemool/Lee/experiments/e14/calibrate_train.py --data kaggle_data/darrow_15min.pkl --surrogate results/e14_audit/direct_7.pkl --out results/e14_audit/calibration_train.json
python Threemool/Lee/experiments/e14/verify_fast.py --data kaggle_data/darrow_15min.pkl --models kaggle_data/darrow_surrogate_gbm.pkl results/e14_audit/direct_7.pkl results/e14_audit/direct_15.pkl --out results/e14_audit/fast_equivalence.json
```

Output directories must be new. Do not overwrite prior model artifacts.
