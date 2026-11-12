# Lee/e19: Temperature trend and predictive-variance continuation

## Motivation before fitting

e18 improves probabilistic scoring and DO frequency but still predicts NH4
5.64 percent versus 17.32 percent observed. Sixteen innovation draws reproduce
the underprediction (5.65 percent), so eight draws alone do not explain it.
Its NH4 scale is at the top of the prespecified grid. This motivates expanding
the training-calibration scale grid and testing a mean-temperature trend.
No development labels are fitted and the e16 adoption gate is unchanged.

Fit-only Ridge(10) on standardized exogenous and air features estimates a
temperature coefficient for NH4 forcing. Restrict it to nonpositive values
and bound its equilibrium slope by the fitting NH4/temperature SD ratio.
Fit monotone GBM forcing after subtracting that linear component, with
temperature excluded from tree splits. Multipliers 0, .5, 1, 2 are fixed in
advance. This extrapolation prior is not a calibrated nitrification equation
or proof of causal temperature response. Other outputs and their bias terms
remain the e17 inner-selected model. Recalibrate NH4 bias only on inner days.
Retain the original nonlinear-temperature model as a fifth reference option.

For every mean option, NH4 innovation scales 0, .5, 1, 1.5, 2 and DO scales
0, .5, 1, 1.5 are selected by the e18 proper inner scoring rule. That is 100
inner combinations. The diagonal state model makes NH4 and DO trajectories
separable, so reuse component trajectories with the same residual-block draws.
Verify the selected assembled trajectory against a full joint rollout.
Only the single inner-selected combination is audited on development, with
16-draw sensitivity and common-innovation air-order checks. A failed choice
does not authorize choosing a different model from development scores.

First 158 dates fit; last 30 training dates calibrate and supply residuals.
All model members, raw inputs, action bounds, horizons and gate semantics
follow e16-e18. Original test dates remain unevaluated. This is exploratory
development on a repeatedly inspected validation set, not a final confirmation.
