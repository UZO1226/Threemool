# Complete fixed tail-innovation result

Source2232e48 completes the unchanged20-setting search, validation and the
tail-specific independent audit. Selection is pit_b95_a0.75. All copied
weights remain identical to e40. No new weight fitting or policy training
occurs. Uniform controls reconstruct the parent. Tail signs, paired indices,
zero NO3 tail, zero-tail e51 day replay and common-noise action order pass.
The independent auditor reconstructs the sampled indices and full sklearn
day and verifies zero-tail e51 validation replay exactly across all cases.

The original joint gate fails DO frequency and member agreement. No member
passes both frequency tolerances. This candidate is not adopted.

| Validation metric | e40 | e51 | e52 tail |
| --- | --- | --- | --- |
| Normalized MAE | .775476 | .796044 | .843997 |
| NH4 MAE, mg/L | 1.977722 | 2.083800 | 1.979642 |
| NH4 predicted frequency, percent | 13.499342 | 16.686842 | 18.790789 |
| DO predicted frequency, percent | .721053 | .963816 | 6.565789 |
| NH4 Brier | .126486 | .128883 | .138671 |
| DO Brier | .013327 | .014217 | .018257 |
| NH4 onsets matched within1h | 3/89 | 10/89 | 7/89 |
| DO onsets matched within1h | 0/25 | 0/25 | 0/25 |

Actual frequencies are17.315789/1.157895 percent. NH4 bias+1.475 points has
date-block interval[-1.907253,5.142467]. DO bias+5.407895 has interval
[3.863141,7.225658], far outside the fixed tolerance. NH4 AP.377925 and
AUC.702602 do not improve over e40. DO AP.010674 is below prevalence.011579,
and AUC.395392 remains poor. No original onset tolerance matches a DO event.

Brier worsens versus e40 for both labels: NH4 difference+.012185,
interval[.005353,.019219], and DO+.004930, interval[.002709,.007235]. Versus
e51, NH4 difference interval crosses zero, while DO difference+.004040 has
interval[.000955,.006849]. DO is also worse than persistence. These intervals
describe reused validation dates, not independent confirmation after many
trials. No favorable setting is substituted after seeing validation.

Physical clip bounds stay unchanged. Among152000 member/draw transitions,
NH4 clips below zero4563 times and above its original upper bound30 times.
DO1/2/3 clip below zero448/8842/6559 times and never above the original bounds.
These are simulated transition counts, not independent dates or policy seeds.
Clipping preserves physical bounds but does not guarantee realistic dynamics.

The observed failure is consistent with the prospectively recorded transport
limitation: calibration contains306/2850 DO violation steps, while validation
has44/3800. Unconditional paired tail magnitudes can transport a prolonged
high-DO regime into different days. This is a plausible explanation, not proof
of a sole cause. The checked implementation paths pass; do not silently shrink
the tail multiplier, change alpha or expand clip bounds to rescue this result.

summary.json and audit.json preserve all settings, saturation, member metrics,
onsets, ranking, references and date uncertainty. artifacts.json records the
draft archive with original checkpoint copies and complete log, remotely
verified by size/SHA256. Paired banks and raw source data are excluded and
recreated by the recorded calibration command. The original final test stays
unused and existing e40 policy comparisons stay frozen.

There is no complete safety-forecast or agent improvement here. Continue the
main objective. e53 separately probes strictly past public observations on
training-only simulated episodes, with a repeated-current control and lower-air
coverage follow-up. That feature diagnostic does not select this environment,
constitute policy learning or resolve the model's DO event limitations.
