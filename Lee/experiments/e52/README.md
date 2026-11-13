# Lee/e52: Retain calibration tail magnitudes outside quantile support

## Prospectively fixed hypothesis

e51 improves frequency point estimates and some NH4 timing but does not improve
DO probability error or onset prediction. Its calibration-only support audit
identifies new DO events outside every member's sampled quantile range.
Reweighting bounded PIT values cannot generate these observed-state events.
Test retaining the signed tail magnitudes that PIT endpoint clipping discards.
This is a new stochastic environment candidate, not an agent extension or an
assertion that any earlier intentional distribution was a coding bug.

Keep frozen e40 weights and original physical clipping bounds. Keep original
158 fit/30 calibration/40 validation/41 unused-test split. As in e51, estimate
one PIT bank per original alpha0,.25,.5,.75 using the actual blended kernel
on inner-calibration observed states. NO3 remains deterministic with PIT.5.

For each member/calibration row, calculate signed target overshoot beyond the
blended lowest/highest conditional quantile. It is zero for outcomes inside
the support, negative below it and positive above it. Keep these four safety
tail innovations paired with that row's five PIT values. Force NO3 innovation
to zero. Draw both arrays using exactly the same original date/member/draw
indices and1/4/16/95-row blocks. Tail magnitudes are calibrated observations,
not new counterfactual plant transitions or fitted controller information.

The transition is alpha*current+(1-alpha)*interpolated_quantile+tail_innovation,
followed by the unchanged physical clip. Add the fixed sampled innovation
before final clipping. Its magnitude is independent of the current action and
future states. Preserve this independence so common-noise action ordering can
still be verified. Nonzero noise changes the stochastic model, not actual
NH4/DO thresholds, rewards, costs or action bounds.

There is one fixed full-strength tail variant, no searched multiplier or
validation-dependent iteration. A zero-tail control must reconstruct e51
exactly, including its selected-bank forecasts. Uniform-noise settings use
zero tail and remain identical to the original parent. Reuse the original20
proper-score settings and selected16-draw sensitivity; PIT settings use their
alpha-matched paired banks. Calibration resubstitution and the large change
in observed DO event prevalence between periods remain limitations.

Before outcome evaluation, require exact zero-tail day replay, deterministic
paired noise indices, correct tail signs/support, finite arrays, zero NO3
innovation, future-test exclusion and reconstruction of clipped calibration
safety targets when using their own paired noise. This last identity is an
algebraic calibration construction, not evidence of generalization. Report
tail occurrence, magnitudes and physical-clip saturation. Do not expand clip
bounds to make calibration targets reconstruct.

After fixed inner selection, report the unchanged e16 gate, all member
frequencies, separate NH4/DO Brier/AP/AUC and onset0/1/4, original/persistence/
e40/e51 references, common-noise air order, independent slow day replay and
10000 five-day paired Brier intervals. Passing aggregate frequency alone
does not establish DO event skill. Preserve failure and all uncertainty. No
automatic environment freeze, favorable e51-cell selection or policy training.

Global calibration tails might transport prolonged high-DO episodes into a
different validation regime and overpredict events. This is a falsifiable
limitation, not a reason to discard a failing seed or shrink innovation after
seeing validation. Conditional or other variants require separate prospective
specification. Five inherited members are not new model or policy seed runs.

Keep raw data, paired banks and derived row arrays outside Git. Compact
scripts/reports stay under Lee/e52; model/log artifacts use the existing
environment draft with verified checksums. Source/license, data hashes, units,
generated-influent and observational-control caveats follow e40/e42. Historical
policy comparisons remain frozen and final test stays unused.

Status: implementation and initial preflight pass. preflight.json records
tail signs/support, calibration algebra, zero NO3 tail, all four block samplers
and exact zero-tail day reconstruction. preflight_air.json additionally checks
the nonzero-tail common-noise air ordering before the full outcome evaluation.
The algebraic calibration replay is not a validation performance result.

```powershell
python Threemool/Lee/experiments/e52/calibrate_tail.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --e51-runs results/e51_blended_pit --preflight-out Threemool/Lee/experiments/e52/preflight_air.json
python Threemool/Lee/experiments/e52/calibrate_tail.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --e51-runs results/e51_blended_pit --out results/e52_tail_innovation
python Threemool/Lee/experiments/e52/audit.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e52_tail_innovation --freeze Threemool/Lee/experiments/e42/environment_freeze.json --e51-runs results/e51_blended_pit --out Threemool/Lee/experiments/e52/audit.json
```

Use fresh output paths when repeating a recorded command. paired_bank.npz
contains both PIT and tail arrays and is excluded from Git. The tail-specific
auditor is required; the older PIT-only auditor would omit the new innovation.
It independently reconstructs sampled indices and a full sklearn day, verifies
the entire zero-tail e51 validation replay, and compares all fixed references.
Statistics pooling calibration members refer to14250 member rows from2850
actual transitions, not14250 independent observations or new policy seeds.
