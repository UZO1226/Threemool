# Lee/e31 joint-horizon results

Execution commit9d4ad6d. Independent step0 alignment, exact target alignment and
finite local-loss gradient checks pass. Five80-epoch models complete. Inner
selection chooses epoch80,n1_d1. The proper score .68472 is worse than e24's
.67460 on the same inner dates, so joint supervision is not established as the
better candidate by its selection criterion.

Development normalized MAE .79429 improves versus e24 .87253. NH4 violation
prediction9.7533 percent underpredicts17.3158 actual by7.5625 points. DO1.9750
overpredicts1.1579 actual by .8171 points, worsening the original reference's
.6421-point error. One of five members meets both tolerances, whereas four
are required. The unchanged adoption gate fails. Air-order checks pass.

The later independent audit checks full-day Torch/NumPy agreement, saved-score
reconstruction, separate event probabilities/timing and date uncertainty.
No policy-training seeds or official test outcomes are used. The failed local
objective is retained. e32 tests persistent pre-episode context, without
combining the two changes or relaxing the adoption criteria.
