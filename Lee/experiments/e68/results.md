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
