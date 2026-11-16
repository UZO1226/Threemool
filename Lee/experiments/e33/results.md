# Lee/e33 kinetic temperature prior results

Execution commit787ca0f. The scaling operator passes15C identity, slower
cold-temperature depletion at fixed coefficients, air-direction and finite
observed-head gradient checks. Full-day Torch/cache agreement also passes.
All five80-epoch fits finish and inner selection chooses epoch80,n1_d1,
proper score .67954. This is worse than e24 .67460 and e32 .64123 on the same
inner dates; the fixed prior is not established as the better environment.

Development normalized MAE .81951 and NH4 MAE1.38278mg/L improve over the original
GBM. NH4 frequency8.5796 percent still underpredicts17.3158 actual by8.7362 points.
DO1.6842 percent differs from1.1579 actual by .5263 points. Zero members meet
the combined safety tolerances. The unchanged gate fails NH4 calibration and
member checks. Recursive common-noise air-order checks pass.

Independent full-day Torch/NumPy and selected-score reconstruction pass.
NH4 frequency-bias date-block95-percent interval[-11.19,-5.96] points remains
substantially negative. DO bias interval[-.203,1.270] is uncertain. Separate
probability, event, reset and actual matrix-norm diagnostics are in audit.json.
The candidate is neither a full ASM1 model nor a Tilburg-calibrated process
model. This trial does not show that all kinetic or mechanistic models fail.
No new policy or official test evaluation. The active goal proceeds to e34.
