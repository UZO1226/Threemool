# Complete quantile exposure result

Sourcee3a0e9d completes105 fits and the unchanged20-setting inner search.
Selection remains pit_b95_a0.5. NO3 refit predictions match frozen parent
probes. Zero-change input/target reconstruction, initial-state agreement,
future-target scrambling, finite/capped training inputs, quantile coordinate
order, fast/sklearn agreement, leaf-function restoration and full-day
independent sklearn reconstruction pass. This run identifies no implementation
fault in these checked paths. Five model members are not policy seeds.

The unchanged adoption gate fails NH4 frequency and member agreement. None
of five members satisfies both frequency tolerances. No model replacement or
controller training follows this result.

| Validation metric | e40 parent | e50 exposure |
| --- | --- | --- |
| Normalized MAE | .775476 | .706653 |
| NH4 MAE, mg/L | 1.977722 | 1.704094 |
| NH4 predicted frequency, percent | 13.499342 | 5.117763 |
| DO predicted frequency, percent | .721053 | .946053 |
| NH4 Brier | .126486 | .144188 |
| DO Brier | .013327 | .013936 |

Actual NH4/DO frequencies remain17.315789/1.157895 percent. NH4 bias is
-12.198026 points with date-block interval[-15.988158,-8.216414]. DO bias
is-.211842 points with interval[-1.057253,.750066]. DO point frequency and
recursive air ordering pass, but they cannot rescue the failed joint gate.
Both common-noise recursive action-order violations are zero.

NH4 probability ranking degrades: AP.367556 versus parent.426755, AUC.693908
versus.716309 and recall at.5.077508 versus.189970. Only2/89 NH4 onsets are
matched within an hour versus parent3/89. DO AP.011521 and AUC.475751 remain
near chance, with no DO onset matched. DO Brier is worse than initial-state
persistence: difference+.002357, paired interval[.000536,.004728].

The NH4 Brier difference from e40 is+.017702, interval[.002202,.033407], worse
on these reused dates despite better MAE. DO difference+.000609 has interval
[-.001307,.002629], crossing zero. Both constraints still improve probability
error over original GBM, but that does not establish improvement over e40 or
accurate rare-event prediction. Date-block intervals are exploratory, not
independent confirmation after many trials.

This falsifies the proposed fixed50/50 augmentation as a complete improvement
here. The augmented forecast state can differ substantially from the actual
state, while its target is still the recorded actual successor. That is a
plausible supervision limitation, not proof of the failure's sole cause.
Doubling safety rows also changes effective leaf support. Without another
control, do not attribute every metric difference exclusively to exposure.
No positive exposure-specific claim needs rescuing with a favorable control.

summary.json, exposure_check.json, audit.json and event_reference_audit.json
retain settings, member results, independent reconstruction, timing, ranking
and uncertainty. artifacts.json identifies the remotely verified existing
draft archive with all five quantile files and complete console log. Raw
inputs and PIT banks are excluded; the original command recreates the bank.
All final test dates remain unused and historical e40 policies stay frozen.

The next alternative keeps e40 weights fixed and checks an explicit sampling
calibration mismatch: PIT is currently estimated for unblended quantiles,
while alpha.5 blends sampled next state with current state. Calibration to
the actual blended one-step distribution has not been tested by this refit.
Specify that separate experiment before running it, retain alpha0 identity,
and do not label the earlier intentional design an implementation bug.
