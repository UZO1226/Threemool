# Result

Source commit: 2dc3dd8. Inner selection chooses pit_b95_a0. No final test was
evaluated and no policy was trained. Validation is repeatedly reused exploratory
data, not an independent confirmation set.

NH4 violation frequency is16.4349 percent versus17.3158 observed. DO frequency
is2.6908 percent versus1.1579 observed. Normalized MAE is.802359 and NH4 MAE
is2.03502 in the prepared dataset's concentration units. The original adoption
gate fails. The closer NH4 frequency does not establish event timing accuracy.

Common-noise recursive action ordering fails: maximum NH4 reversal5.41028 and
DO reversal.268395. Fast inference exactly matches sklearn, so acceleration
does not explain the failure. The nominal monotonic constraints were not
preserved by the installed quantile leaf updates. e40 independently reproduces
this behavior on synthetic data and tests a scoped correction. Retain this run
as the unchanged implementation control rather than retrospectively replacing
its saved results. See summary.json for all member and event diagnostics.

The independent full-day sklearn inference and NumPy interpolation audit
reconstructs the selected forecast. Date-block95 percent intervals for NH4
frequency bias are[-3.77786,1.99095] percentage points and DO bias
[.17171,3.25789]. These intervals use exploratory validation dates. They do not
repair the action-order failure or establish policy improvement. See audit.json.
