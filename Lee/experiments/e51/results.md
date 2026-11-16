# Blended-kernel PIT result

Source1a2634d completes the fixed20-setting search with no weight fitting.
All eight unchanged inner controls reproduce parent metrics within required
numeric tolerance. Five copied checkpoint hashes match e40 exactly. Kernel
preflights and independent full-day sklearn reconstruction pass. Initial
NO3 clip-bound assumption failure and its correction remain recorded; no data,
parent weights, clip limits or acceptance criteria were changed.

Inner selection is pit_b95_a0.25. The unchanged e16 point gate passes every
check, with all five members within both frequency tolerances. This is a
point-frequency result, not a complete safety-forecast or policy improvement.

| Validation metric | e40 parent | e51 selected |
| --- | --- | --- |
| Normalized MAE | .775476 | .796044 |
| NH4 MAE, mg/L | 1.977722 | 2.083800 |
| NH4 predicted frequency, percent | 13.499342 | 16.686842 |
| DO predicted frequency, percent | .721053 | .963816 |
| NH4 Brier | .126486 | .128883 |
| DO Brier | .013327 | .014217 |
| NH4 new onsets matched within1h | 3/89 | 10/89 |
| DO new onsets matched within1h | 0/25 | 0/25 |

Actual NH4/DO frequency is17.315789/1.157895 percent. NH4 bias-.628947
points has date-block interval[-4.543618,3.457911]; DO bias-.194079 has
interval[-1.325674,1.159901]. The latter interval extends beyond1 point,
so passing point criteria does not prove a population calibration guarantee.

NH4 AP.411940 versus e40.426755 declines, while AUC.721151 versus.716309
increases slightly. Recall at.5 is.205167 versus.189970. Exactly1 NH4 onset
matches versus0 before; within one hour10 match, with most89 still missed.
DO AP.011710 is near prevalence.011579 and AUC.480338 remains near chance.
No DO onset matches at any original tolerance. Do not claim DO onset is solved.

NH4 Brier difference+.002397 has paired-date interval[-.007196,.013252],
crossing zero. DO difference+.000890 has interval[.000002956,.002084725],
worse on these reused dates. The very small lower endpoint and many prior
trials prohibit independent-confirmation or population-significance claims.
DO is also worse than initial-state persistence: difference+.002638,
interval[.000136,.005910]. Probability error improves over original GBM for
both labels, but does not improve over the frozen e40 for both labels.

This is a partial frequency/timing change, not a complete replacement of e40.
Historical policy studies stay frozen. No agents are trained on this candidate.
Five inherited model members are not five new model fits or five policy seeds.
Final test remains unused, and there is no external blocker to further work.

The separate calibration-only support audit finds that alpha.5 cannot express
9/26 observed new DO events in any member's conditional support. PIT
reweighting cannot create outcomes outside that support for observed-state
prediction. This is not a bound on free-running forecasts. See support_results.md.

The post-primary fixed2x2 bank/alpha decomposition completes under source58253c4.
Both original primary cells reconstruct their reports and the primary summary
hash stays unchanged. No cell replaces the selected candidate.

| Bank and alpha | NH4 frequency, percent | DO frequency, percent | NH4 Brier | DO Brier |
| --- | --- | --- | --- | --- |
| Original,.25 | 15.542105 | 1.006579 | .125543 | .014446 |
| Original,.5 | 13.499342 | .721053 | .126486 | .013327 |
| Matched,.25 | 16.686842 | .963816 | .128883 | .014217 |
| Matched,.5 | 16.303947 | .675658 | .132860 | .013138 |

At fixed alpha.25, matched-minus-original Brier difference is+.003339 for
NH4, interval[-.000805,.008057], and-.000229 for DO, interval[-.000660,.000017].
At fixed alpha.5 both intervals also cross zero. No clear probability-error
benefit of the PIT change alone is established. Changing original-bank alpha
from.5 to.25 increases DO Brier by+.001119, interval[.000026,.002585]. All
these after-outcome contrasts are exploratory and unadjusted for multiplicity.
They separate two changed factors but do not establish a broadly superior
calibration method. Every cell and contrast remains in decomposition.json.

summary.json, audit.json and event_reference_audit.json retain the complete
metrics, member frequencies, uncertainty, checks and selection. Model/log
artifacts are archived separately without raw data or PIT arrays; regenerate
the bank using the recorded calibration command. All commands and the initial
preflight failure remain in the experiment record.

Next investigate the tail-support limitation measured on inner calibration:
PIT endpoint values currently return boundary quantiles rather than actual
out-of-support concentrations. A separately specified tail-innovation
candidate can retain those magnitudes with fixed, action-independent noise
and unchanged physical clip bounds. It must preserve zero-tail identity,
common-noise action ordering and the same selection/gate, and must not use
these decomposition cells to choose a favorable validation setting.
