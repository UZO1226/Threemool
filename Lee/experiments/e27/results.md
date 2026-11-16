# Lee/e27 state-memory-cap results

Execution commit17fac66. The initialization comparison passes: zero-temperature
coefficients match e24, equilibrium matches, and all observed heads have nonzero
gradients. All five models complete80 epochs. Inner selection chooses epoch80,
n1_d1. Development normalized MAE is .83329 and NH4 MAE1.36676mg/L, compared
with e24 .87253 and1.51352. NH4 violation prediction is8.6401 percent against
17.3158 actual. DO prediction is1.7145 against1.1579 actual. Zero members meet
the combined safety tolerances; the unchanged environment gate fails.

No recursive air-order violations occur in the common-noise probes. Independent
Torch/NumPy full-day forecasts and saved-score reconstruction pass. The measured
maximum standardized state row sum is .9928165, with zero rows above1 on the
development cache. Thus the larger allowed cap was not exercised as amplifying
dynamics. The result does not establish the proposed amplification mechanism.

Five-day date-block95-percent intervals give NH4 frequency bias[-11.26,-5.86]
percentage points and DO bias[-.088,1.220]. NH4 recall at fixed probability .5
is .2401. Full event diagnostics, uncertainty and reset checks are in audit.json.
No policy seeds or test outcomes are included. Inner calibration remains reused
for bias and selection; this is exploratory rather than independent confirmation.

The selected checkpoint is at the training endpoint and fitting traces continue
declining. e29 therefore continues fitting, including this checkpoint as an
unchanged control. It uses the explicit SlowMemoryModel when loading weights:
loading the same state dictionary into the base .995-cap class would silently
change the transition rule, so audit.py now checks model kind against metadata.
