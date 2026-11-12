# Lee/e30 rare-event regression results

Execution commit76d4c86. The independent check confirms exactly4:1 rare/ordinary
regression gradients at equal errors and ordinary-loss equivalence without
events. Five80-epoch fits finish with finite parameters. Inner selection chooses
epoch80,n1_d1. Proper score .69771 is worse than e24's .67460 on the same inner
dates; weighting is not selected as a superior environment.

Development normalized MAE is .85053. NH4 violation prediction6.7493 percent
underpredicts17.3158 actual by10.5664 points. DO prediction2.3980 overpredicts
1.1579 actual by1.2401 points. All five members fail the combined safety check;
the unchanged gate fails NH4 and DO calibration. Action-order checks pass.
This is not a treatment-quality improvement and no policy is trained on it.

The weighting fails to recover NH4 event frequency and increases DO frequency
too much. Exact event timing, Brier and uncertainty are in audit.json. A lower
aggregate prediction MAE is insufficient for adopting this model. Next e31
tests missing local transition supervision while retaining the original full-day
loss, instead of retaining these tail weights. Official test remains unused.
