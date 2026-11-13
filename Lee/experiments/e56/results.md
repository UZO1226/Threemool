# Four-step rate correction fails the complete improvement criterion

The entire source0054e89 study finishes: both horizons for policy seeds0..4,
twenty scalar regressors, ten held-model audits, ten native/matched comparisons,
ten independent action audits, both five-seed reports and the horizon contrast.
All base-policy checkpoint/config/source/budget guards pass. The original
401280-step policies and captured streams are reused without PPO retraining
or any extra environment-training transition. No final test is evaluated.

| Matched-air mean | Original FCSRL | Primary horizon4 | Regression control horizon1 |
| --- | --- | --- | --- |
| Air, m3 per23.75h | 52020.577481 | 52004.114677 | 52003.329142 |
| Summed violation cost | 22.464125 | 22.422125 | 22.503000 |
| NH4 violation, percent | 23.208684 | 23.242500 | 23.321053 |
| DO violation, percent | .437763 | .359737 | .366316 |

Every actual air match passes. Primary horizon4 cost difference-.042000 has
paired-policy-seed95 interval[-.103375,.011875], and NH4 difference+.033816
percentage points has interval[-.054474,.156447]. Both fail their predefined
conditions. DO difference-.078026 points has interval[-.151974,-.028289],
passing only the DO condition. Thus the primary fails complete adoption.

The horizon1 regression control also fails. Its cost difference+.038875 has
interval[-.043750,.163750], while NH4+.112368 points has interval
[.016842,.291184], favoring the original FCSRL under this exploratory comparison.
DO-.071447 points has interval[-.123158,-.031579]. Do not substitute this
control for the failed primary or choose a favorable seed.

The secondary same-base/capture/reference-air contrast verifies identical base
checkpoints and training streams, and mutual air matching. Horizon4 minus
horizon1 cost difference-.080875 has interval[-.158750,-.020000], NH4-.078553
points has interval[-.137500,-.021842], and DO-.006579 points has interval
[-.029868,.008553]. The longer horizon improves cost/NH4 relative to its
matching regression control, but the DO non-worsening condition is uncertain.
This is not a complete horizon-specific benefit and does not establish an
improvement over original FCSRL. In particular, mitigating the control's NH4
harm does not make the primary better than the original reference.

Both native policies use slightly less air, so their native quality figures
are also preserved. Primary native air51987.800250m3, cost22.425000,
NH423.250526 percent and DO.354737 percent still fail cost/NH4 conditions.
The control's native NH4 interval is also unfavorable. Matching is applied
after the corrected action using the unchanged air-only selector and original
query/tie/tolerance rules; no quality result selects an offset.

The first52 captured updates fit each model; the last14 diagnose it. All
physical labels, episode/reset/lane/update boundaries, public inputs and
serialized predictions are verified. Raw/clipped fast inference matches
sklearn, tested coordinate-order violation is zero, and every actual native
action is independently reconstructed. The constant-rate null reproduces the
original full-day smoke. No implementation fault is identified by these
checks; this does not prove that every possible fault is absent.

The added proposal diagnostic uses held training states only. Many predicted
action-risk curves are flat and the fixed penalty suppresses most risk-only
changes. This explains algorithmic activation, not true counterfactual safety.
Risk-only alternatives are not evaluated controllers, and their apparent
prediction gain does not justify changing the current penalty. Longer-horizon
labels also average future stochastic behavior-policy actions from changing
PPO updates; they need not estimate the final deterministic controller's
counterfactual costs. These are possible limitations, not proven sole causes.
Lower MSE across different horizon labels cannot establish better prediction.

The e55 classifier differs in loss and retained NO3 information, so cross-family
differences cannot be attributed to horizon alone. Frozen e40 DO onset/ranking,
surrogate counterfactual and season-shift limits persist. All intervals use
five original policy seeds, sample SD ddof1 and10000 paired-seed resamples.
Model/world/noise draws, dates and regressor seeds are not extra policy seeds.
Repeated validation and multiple exploratory trials are not independently
confirmed or multiplicity-adjusted evidence. Preserve all failed components,
native results and uncertainty. No rule, original cost or test split changes.
This failed complete criterion does not complete the active goal.

artifacts.json verifies lee-e56-rate-correction.zip,15,403,571 bytes, SHA256
072be5a2fef3b64c52e447166e07d65bc9d797cc746a3590d3b4442537f3ece1,
in private policy draft443333191, asset743519310. Both horizons' models, complete
logs, all generated evaluations/queries/audits and final reports are retained,
with the five unchanged final base checkpoints and required source/config
manifests. Source observations, captured training arrays and stochastic banks
are excluded. Archive/internal CRC and hashes and remote size/digest verify.
No e56 job remains queued. A distinct fixed executed-air smoothing hypothesis
can test a simple actuator mechanism without refitting a logged-cost predictor;
specify its gain and reset/offset/identity checks before any outcomes.
