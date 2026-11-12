# Lee/e18: Probabilistic residual innovations for safety frequencies

## Motivation and fixed scope

e17's inner-selected n95_d20_cal has NH4 MAE 1.746 versus 2.482 for the old
model but predicts only 4.16 percent NH4 violations versus 17.32 percent
observed; DO violation is zero. It fails the unchanged e16 adoption gate.
Better point forecasts need not represent the distribution of constraint
events. This motivates a probabilistic innovation model, not a relaxed gate.

Use only the e17 inner-selected deterministic model. Build member-specific
teacher-forced one-step residual banks from the same 30 training calibration
days. Center every output's residuals and keep their cross-output dependence.
NO3 innovations are set to zero to avoid an unrelated variance change.
Draw four-step residual blocks with fixed random seeds, eight paths per model
member. Never read development residuals to fit or sample innovations.
The common innovations are reused across scale candidates and paired air
interventions. As innovations do not depend on actions, the conditional
recursive action ordering and initial-state contraction remain intact.

## Prespecified selection and verification

NH4 innovation scales: 0, .25, .5, 1. DO scales: 0, .5, 1, 1.5. The 16-point
grid is selected solely on inner calibration. Selection score is normalized
empirical CRPS averaged across five concentrations plus NH4 and DO Brier
scores. It scores the full predictive distribution, rather than rewarding a
small mean error by suppressing variance. Evaluate every grid point on inner
data, and only the selected point and zero-noise reference on development.
No scale tuning on development outcomes.

The original e16 adoption gate remains applied to mean absolute trajectory
errors, calibration errors, action order and at least four of five structural
members. Average noise paths within each structural member for its frequency
check; noise paths do not count as additional model members or policy seeds.
Also report point-mean errors, CRPS, Brier scores, and noise-path sensitivity.
A candidate that fails is not used for new policy learning.

Verify that zero innovation equals deterministic prediction, empirical CRPS
equals the pair-distance formula on a small sample, intervention ordering
holds with common innovations, and initial-state contraction is preserved.
Original observations and residual banks remain outside Git. Models are
still observational; this procedure cannot validate real intervention effects.
Test remains unused. Fit/calibration/development dates and runtime follow e17.

The scoring-rule motivation follows Gneiting and Raftery, Strictly Proper
Scoring Rules, Prediction, and Estimation, JASA 102 (2007), 359-378:
https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
