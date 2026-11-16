# Lee/e36 proper-score shared calibration results

Execution commit2fabc8a. Frozen e34 weights and noise pass SHA256 checks and
unchanged-parent score reconstruction on inner and development dates. The
coordinate search reduces inner proper score .6244246 to .6219964. This
optimization is not independent evidence of improved forecast generalization.

Selected biases are[-.15625,.09484679,.071875,.00938676,.0671875] in output order.
NO3 bias is unchanged. Development normalized MAE .75517 improves slightly
from parent .76011, but NH4 frequency falls from9.1053 to7.6638 percent against
17.3158 actual. DO1.5099 versus1.1579 actual has error .3520 points. Zero members
meet both safety tolerances and the unchanged adoption gate fails. Air order
passes. Thus matching the calibration objective alone does not fix the failure.

Independent full-day Torch/NumPy and selected-score reconstruction pass using
--parent-runs to load the frozen parent and the new bias. NH4 bias date-block
95-percent interval[-12.42,-6.51] points remains large. Other probability,
timing, horizon and uncertainty diagnostics are in audit.json. There are no
new fitting seeds, policy seeds, protocol edits or final test evaluations.
