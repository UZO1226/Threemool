# Lee/e26 correlated-noise results

Execution commit 629cc6c. The original four-step reconstruction passes exactly.
Among35 inner candidates, a whole95-step NH4 residual block at scale1 wins:
proper score .66859 versus .67460 for e24. The scale is not at the expanded
upper boundary. Thus larger amplitude alone does not solve this calibration.

Development normalized MAE is .88372 versus .87253 for e24. NH4 violation
prediction rises from5.87 to6.86 percent, still below17.32 actual by10.45
percentage points. None of five members meets the safety gate. DO remains
1.2671 percent, unchanged because only NH4 noise timing changes. Action-order
violations remain zero. Sixteen-draw metrics are preserved in summary.json.

NH4 event probability Brier improves .12237 to .11964, but recall at probability
.5 is only .2021: 133 true positives and525 missed event steps. Precision is
.9301. These metrics are descriptive at a fixed threshold, not a threshold
selected from development outcomes. Predicted sample trajectories average
.7825 NH4 runs/day versus2.5 observed, with mean duration8.33 versus6.58 steps.
The frequency error includes missing events rather than only a threshold offset.

DO has no true positives at probability .5, despite a close aggregate frequency.
Its Brier remains .01272. Full reliability bins, runs, and counts are retained,
so a matching mean frequency is not misreported as accurate event localization.
No new agent training or test evaluation. The active goal continues with e27's
state-memory-cap ablation and e28's within-training seasonal partition.
