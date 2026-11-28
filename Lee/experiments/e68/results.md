# Actual fresh policy baselines, seeds5 through9

All five original FCSRL training runs and their passive-capture twins finish
401280 steps. Every final checkpoint file and stored native metric is exactly
equal within each original/captured pair. These are five actual independent
policy seeds, not ten seeds. All fixed horizon4 risk models finish training,
with original sklearn equivalence and monotonic-coordinate checks passing.
The legacy one-step preparation is preserved for provenance and is not used
by the horizon4 controller.

For every actual seed, unchanged e63 reconstruction completes native replay,
air-only matching to that seed's original FCSRL, and independent native/matched
sklearn action and full trajectory audits. Each replay contains152000 actions.
All five actual-air matches pass. The primary fresh e63 reference includes
its fixed post-wrapper offset. It is not the raw NN or e63 native policy.
baseline_authentication.json binds seed IDs, checkpoints, horizon4 model,
configuration, fixed offset, full metrics and audit hashes. There is no
candidate evaluation or candidate performance conclusion in this experiment.

training_summary.json and rate_preparation_summary.json retain compact final
training evidence. Runtime manifests, final model weights, complete logs,
queries and reference audits are in the existing private draft release:

https://github.com/UZO1226/Threemool/releases/tag/lee-e42-e44-policy-comparison-20261113

Asset757595492, lee-e68-fresh-baselines.zip,26658731 bytes,
SHA256 e03b7b321f8d91df7e3a6fb3da8bed6d023ef5832b25c7e694d460eaf97b01d6.
artifacts.json records individual file hashes and verified remote size/digest.
Draft assets require authorized repository access. Raw observations, capture
arrays, residual/PIT banks and intermediate NN checkpoints are excluded.

Reproduce run_baselines.py, prepare_rates.py and run_references.py in that
order with the frozen e42 dependencies and original data restored by hash.
Exact executed source identities and commits are retained in archive manifests.
Use a new workspace/output root: existing completed outputs are immutable.
package_fresh_baselines.py independently checks all terminal statuses, both
checkpoint copies, risk-model hashes and all reference audits before archiving.
Controls confirm that incomplete training and a failed native trajectory
audit are rejected before any archive is written.

e71 must still pass its own separate fresh comparison and all independent
audits. Reused validation40 remains exploratory and final test41 is preserved.

## Independent held logged-action errors

The original e56/audit_rates.py now independently reconstructs the actual
85120 held transitions per seed, updates53 through66, without refitting.
Both horizon4 regressors have lower rate MSE than current-state persistence
and the fit-mean constant on every actual fresh seed. Each persistence
contrast's paired-date95% interval is below zero. These are conditional
date resamples,10000 with RNG5602, not policy-seed improvement intervals.

| Seed | NH4 rate MSE | NH4 difference from persistence | DO rate MSE | DO difference from persistence |
| --- | --- | --- | --- | --- |
| 5 | .004265317 | -.002938565 | .003375695 | -.005945623 |
| 6 | .003900696 | -.002764894 | .004880406 | -.005843996 |
| 7 | .003684566 | -.003130894 | .005770597 | -.008916512 |
| 8 | .004926019 | -.003600424 | .001102109 | -.002295545 |
| 9 | .001845701 | -.001604657 | .006829244 | -.008415656 |

held_audit_seed files and held_artifacts.json retain all labels, controls,
date counts, intervals and hashes. Held update blocks are excluded from risk
model fitting and projection calibration; they are not held out from the base
NN's training, and their dates overlap fitting dates. These diagnostics do
not measure candidate actions' counterfactual water quality or repair the
separate recursive environment's onset forecasts.

The additional archive is lee-e68-fresh-held-audits.zip,57594 bytes,
SHA256 6cfb4d75f93715322222777e1d7d124a3febfdda888604cfdfd9146e58dec422,
existing draft asset757618888. Remote size and digest are verified. Its first
local packaging attempt exposed Windows ZIP path normalization, failed before
upload, and is retained locally with a failed-windows-names suffix. The corrected
archive uses POSIX entry names and passes every stored-byte integrity check.

Reproduce each seed's original held audit, substituting actual IDs5 through9:

```powershell
python Threemool/Lee/experiments/e56/audit_rates.py --run results/e68_fresh_baselines/captured/new_ppolag_fcsrl/seed_5 --baseline results/e68_fresh_baselines/original/new_ppolag_fcsrl/seed_5 --freeze Threemool/Lee/experiments/e42/environment_freeze.json --data kaggle_data/darrow_15min.pkl --correction results/e68_fresh_rates/models/h4/seed_5 --out results/e68_fresh_held_audits/seed_5.json --horizon 4 --phase held
```

package_fresh_held_audits.py authenticates all five outputs, their fit-summary
metrics, model hashes and original audit-source hashes before archiving.
