# Lee/e15: Risk-stratified FCSRL training after environment audit

This is an exploratory personal experiment, not a change to Compare/PROTOCOL v1.
It uses the e14 selected model. No test dates are evaluated: PROTOCOL reserves
them for the team's one final comparison. The selection statement in the initial
e14 plan is superseded by that existing team rule.

## Frozen hypotheses and settings

Compare PPO-Lag, FCSRL + PPO-Lag, and FCSRL with risk-stratified training resets.
The extension changes only training-day sampling. Rank training days by observed
NH4 violation frequency and sample the highest-risk quarter with probability
one half, the remaining days with probability one half. Ties use date order.
The hypothesis is that exposure to difficult days reduces validation NH4
violations under the seasonal shift. This is a sampling extension, not a new
FCSRL representation loss, and no originality claim is made.

All methods use seeds 0, 1, 2, 3, 4, 401280 transitions, 8 environments,
760-step rollouts, 66 updates and five validation checkpoints. Reward, action,
state and cost remain e07 definitions. Calibrate the daily training cost limit
using training replay in the frozen environment. Compare final policies as
required by v1, with selected checkpoints as secondary results. Always report
NH4 and DO violations separately and air volume over 23.75 hours, not measured
electricity or a full 24-hour total.

Primary extension comparison is seed-paired final cost and NH4 violation,
with air consumption, DO violation, standard deviations and all seed results.
Report an improvement only if mean cost and NH4 violation decrease while air
increases at most 5 percent and DO violation does not increase more than one
percentage point. Report paired 95 percent bootstrap intervals (10000 draws,
fixed analysis seed 2026); intervals spanning zero do not establish improvement.
The 5 percent threshold is an experimental preference, not a universal rule.
Failure and energy-safety trade-offs are preserved. Five seeds are exploratory.

Environment selection is based on prediction metrics in e14. Controller results
only compare methods within that frozen environment, and cannot establish
real-plant performance or isolate gains due to changing the environment.

## Execution notes

The first run at results/e15_runs was interrupted by Windows refusing an
atomic progress.json replacement while the runner was reading progress.
No completed run metrics were produced. All partial artifacts are preserved
and excluded. Bounded retries for transient PermissionError were added to
training and runner JSON writes. The full comparison restarts from scratch
at results/e15_runs_retry, with unchanged learning settings and random seeds.
restart_verification.json confirms that all five restarted PPO runs reproduce
their interrupted logs exactly through the common 56 to 61 update prefixes.

Frozen final policies are also evaluated without retuning on the old delta
GBM and direct 15-leaf validation simulators. These are model sensitivity
checks, not independent real-world validation. No test dates are used.

A further frozen diagnostic evaluates baseline final policies with normalized
action biases -0.2, -0.1, 0, 0.1, 0.2, clipped to the same action bounds.
This asks whether simply supplying more air explains a cost difference.
Interpolate the resulting five mean points only inside their observed air
range. This is a fixed-policy sensitivity curve, not a retrained Pareto
frontier, and is not used to select or alter the extension. Zero bias must
reproduce the saved final-policy metrics.

## Reproduction

Run from the parent directory of Threemool using the project environment:

```powershell
python Threemool/Lee/experiments/e15/run.py --data kaggle_data/darrow_15min.pkl --surrogate results/e14_audit/direct_7.pkl --calibration results/e14_audit/calibration_train.json --upstream FCSRL --out results/e15_runs_retry --parallel 5
python Threemool/Lee/experiments/e15/cross_model.py --runs results/e15_runs_retry --data kaggle_data/darrow_15min.pkl --models kaggle_data/darrow_surrogate_gbm.pkl results/e14_audit/direct_15.pkl --out Threemool/Lee/experiments/e15/cross_model.json
python Threemool/Lee/experiments/e15/air_sweep.py --runs results/e15_runs_retry --data kaggle_data/darrow_15min.pkl --model results/e14_audit/direct_7.pkl --out Threemool/Lee/experiments/e15/air_sweep.json
python Threemool/Lee/experiments/e15/plot.py --runs results/e15_runs_retry --summary Threemool/Lee/experiments/e15/summary.json --audit Threemool/Lee/experiments/e14/summary.json --out Threemool/Lee/experiments/e15/figures
python Threemool/Lee/experiments/e15/package_artifacts.py --runs results/e15_runs_retry --audit results/e14_audit --old-model kaggle_data/darrow_surrogate_gbm.pkl --out results/lee_e14_e15_artifacts.zip --index Threemool/Lee/experiments/e15/artifacts.json
```

Use a new output path for a rerun. run.py enables verified fast dynamics and
one thread per numerical library. Runtime package versions and sampling
diagnostics are in runtime.json. Checkpoints and detailed logs remain local
at results/e15_runs_retry; their data and model hashes are in the manifest.
They can be regenerated with these commands. Source commit and per-file
hashes are recorded by the existing e07 runner. No upstream FCSRL file is
modified (d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0).

The extension changes the training distribution and therefore the meaning of
training-average cost; uniform validation comparison is essential. It does
not provide a formal safety guarantee. Existing cost values 0, 1, 2 and DO
upper-limit semantics are retained for comparability and are not legal limits.
