# Lee/e29 continuation results

Execution commit2cd6946. All five models complete240 additional epochs, total320,
with finite parameters. Continued fitting losses decrease, but the original
warm-start checkpoint wins the inner comparison. After unchanged bias calibration,
proper scores at total80/160/320 are .723069/.738157/.763832. The chosen additional
epoch is0, total80. Selected noise remains n1_d1, and the selected development
forecast exactly retains e27's failing result. This is a failed optimization
continuation, not evidence that a newly trained model beats e27.

The lower training loss and higher inner error do not support more epochs of
this same objective as the next action. They also do not prove every neural
model is optimized or that the dataset cannot support a controller. e30 changes
the environment regression objective toward rare safety events, preserving the
proper inner selection and all adoption thresholds. No official test or new
policy-training seed is used here. Full traces and checkpoint hashes are in
summary.json; additional independent reconstruction and event checks use the
explicit slow-cap model in audit.json.
